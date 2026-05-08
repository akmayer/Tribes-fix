"""
PyTorch model architecture for Tribes policy and value prediction.

The model takes a game state (JSON payload from Java bridge) and outputs:
1. Factorized action logits: [action_type_logits, source_logits, target_logits, param_logits]
2. State value estimate for training

State Representation:
- Board: 11x11 grid encoding terrain, resources, buildings, unit presence
- Units: Feature vectors for each active tribe unit
- Cities: Feature vectors for each active tribe city
- Tech tree: One-hot encoded researched technologies
- Tribe stats: Stars, score, population, etc.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import numpy as np
from typing import Dict, Tuple, Optional
from pathlib import Path


class StateEncoder:
    """Encodes game state JSON payload to tensors suitable for NN input."""
    
    def __init__(self, board_size: int = 11, max_units: int = 100, max_cities: int = 50):
        self.board_size = board_size
        self.max_units = max_units
        self.max_cities = max_cities
        
        # Feature dimensions (will be flattened into state vector)
        # Board: 11x11 with 4 channels (terrain one-hot, resource, building, unit presence)
        # For simplicity, we'll flatten the board and unit/city features
        self.board_features = board_size * board_size * 8  # 8 channels per tile
        self.unit_features = max_units * 16  # 16 features per unit (type, health, position, etc.)
        self.city_features = max_cities * 10  # 10 features per city (level, population, position, etc.)
        self.tech_features = 50  # Technology tree (one-hot over ~50 techs)
        self.tribe_features = 10  # Tribe stats (stars, score, population, etc.)
        
        self.total_state_size = (
            self.board_features + 
            self.unit_features + 
            self.city_features + 
            self.tech_features + 
            self.tribe_features
        )
    
    def encode_board(self, board_dict: Dict) -> np.ndarray:
        """
        Encode board into tensor.
        
        Each tile gets 8 channels:
        - Terrain (one-hot over ~8 terrain types)
        - Resource presence (binary)
        - Building presence (binary)
        - Unit presence (binary)
        - Etc.
        """
        size = board_dict.get("size", self.board_size)
        board_tensor = np.zeros((size, size, 8), dtype=np.float32)
        
        # Terrain type mapping (simplistic; should match Types.java TERRAIN enum)
        terrain_map = {
            "WATER": 0, "SHALLOW_WATER": 1, "GRASS": 2, "FOREST": 3,
            "MOUNTAIN": 4, "SNOW": 5, "DESERT": 6, "CITY": 7
        }
        
        resource_map = {
            "STARS": 1, "CUSTOM": 2, None: 0
        }
        
        building_map = {
            "MONUMENT": 1, "TEMPLE": 2, "ROAD": 3, None: 0
        }
        
        tiles = board_dict.get("tiles", [])
        for x_row in tiles:
            for y_tile in x_row:
                x, y = y_tile.get("x"), y_tile.get("y")
                if 0 <= x < size and 0 <= y < size:
                    # One-hot terrain
                    terrain_idx = terrain_map.get(y_tile.get("terrain", "GRASS"), 2)
                    board_tensor[x, y, terrain_idx] = 1.0
                    
                    # Resource indicator
                    resource = resource_map.get(y_tile.get("resource"), 0)
                    board_tensor[x, y, 5] = float(resource > 0)
                    
                    # Building indicator
                    building = building_map.get(y_tile.get("building"), 0)
                    board_tensor[x, y, 6] = float(building > 0)
                    
                    # Unit present
                    board_tensor[x, y, 7] = 1.0 if y_tile.get("unit_id", -1) != -1 else 0.0
        
        return board_tensor.flatten()
    
    def encode_units(self, active_tribe_id: int, tribes_list: list) -> np.ndarray:
        """Encode unit features for active tribe."""
        unit_tensor = np.zeros(self.max_units * 16, dtype=np.float32)
        
        unit_type_map = {
            "WARRIOR": 0, "ARCHER": 1, "KNIGHT": 2, "DEFENDER": 3,
            "SWORDSMAN": 4, "RIDER": 5, "CATAPULT": 6, "BOAT": 7,
            "SHIP": 8, "BATTLESHIP": 9, "MIND_BENDER": 10
        }
        
        unit_idx = 0
        for tribe_dict in tribes_list:
            if tribe_dict.get("tribe_id") != active_tribe_id:
                continue
            
            for unit in tribe_dict.get("units", []):
                if unit_idx >= self.max_units:
                    break
                
                # Extract features (16 per unit)
                unit_type = unit_type_map.get(unit.get("type", "WARRIOR"), 0)
                health_pct = unit.get("current_hp", 1) / max(unit.get("max_hp", 1), 1)
                position = [unit.get("x", 0) / self.board_size, unit.get("y", 0) / self.board_size]
                atk_val = unit.get("atk", 1) / 5.0
                def_val = unit.get("def", 1) / 5.0
                mov_val = unit.get("mov", 1) / 3.0
                
                offset = unit_idx * 16
                unit_tensor[offset + 0] = float(unit_type) / 10.0
                unit_tensor[offset + 1] = health_pct
                unit_tensor[offset + 2] = position[0]
                unit_tensor[offset + 3] = position[1]
                unit_tensor[offset + 4] = atk_val
                unit_tensor[offset + 5] = def_val
                unit_tensor[offset + 6] = mov_val
                unit_tensor[offset + 7] = 1.0 if unit.get("has_moved") else 0.0
                unit_tensor[offset + 8] = 1.0 if unit.get("has_attacked") else 0.0
                unit_tensor[offset + 9] = 1.0 if unit.get("is_veteran") else 0.0
                unit_tensor[offset + 10:16] = 0.0  # Padding
                
                unit_idx += 1
        
        return unit_tensor
    
    def encode_cities(self, active_tribe_id: int, tribes_list: list) -> np.ndarray:
        """Encode city features for active tribe."""
        city_tensor = np.zeros(self.max_cities * 10, dtype=np.float32)
        
        city_idx = 0
        for tribe_dict in tribes_list:
            if tribe_dict.get("tribe_id") != active_tribe_id:
                continue
            
            for city in tribe_dict.get("cities", []):
                if city_idx >= self.max_cities:
                    break
                
                level_norm = city.get("level", 1) / 5.0
                population_norm = city.get("population", 1) / 50.0
                position = [city.get("x", 0) / self.board_size, city.get("y", 0) / self.board_size]
                is_capital = 1.0 if city.get("is_capital") else 0.0
                has_walls = 1.0 if city.get("has_walls") else 0.0
                
                offset = city_idx * 10
                city_tensor[offset + 0] = level_norm
                city_tensor[offset + 1] = population_norm
                city_tensor[offset + 2] = position[0]
                city_tensor[offset + 3] = position[1]
                city_tensor[offset + 4] = is_capital
                city_tensor[offset + 5] = has_walls
                city_tensor[offset + 6:10] = 0.0  # Padding
                
                city_idx += 1
        
        return city_tensor
    
    def encode_tech_and_tribe(self, active_tribe_id: int, tribes_list: list) -> np.ndarray:
        """Encode technology tree and tribe stats."""
        features = np.zeros(self.tech_features + self.tribe_features, dtype=np.float32)
        
        for tribe_dict in tribes_list:
            if tribe_dict.get("tribe_id") != active_tribe_id:
                continue
            
            # Tech tree (simplified: just use first 50 elements if available)
            tech_data = tribe_dict.get("technology", {})
            if isinstance(tech_data, dict) and "researched_techs" in tech_data:
                # Binary encoding of researched techs
                for i, tech_researched in enumerate(tech_data.get("researched_techs", [])[:self.tech_features]):
                    features[i] = 1.0 if tech_researched else 0.0
            
            # Tribe stats (stars, score, etc.)
            stars = min(tribe_dict.get("stars", 0) / 100.0, 1.0)
            score = min(tribe_dict.get("score", 0) / 1000.0, 1.0)
            
            features[self.tech_features + 0] = stars
            features[self.tech_features + 1] = score
            features[self.tech_features + 2:] = 0.0  # Padding
            
            break
        
        return features
    
    def encode(self, payload: Dict) -> torch.Tensor:
        """
        Encode full game state payload to a single state tensor.
        
        ⚠️ WARNING: This encoder does NOT enforce fog-of-war filtering.
        It encodes the full board as provided in the payload.
        The payload from PythonBridge MUST filter enemy units/cities
        by observability grid before sending here.
        
        Args:
            payload: JSON payload from Java PythonBridge
        
        Returns:
            State tensor of shape (state_size,)
        """
        board_features = self.encode_board(payload.get("board", {}))
        unit_features = self.encode_units(payload.get("active_tribe_id"), payload.get("tribes", []))
        city_features = self.encode_cities(payload.get("active_tribe_id"), payload.get("tribes", []))
        tech_tribe_features = self.encode_tech_and_tribe(payload.get("active_tribe_id"), payload.get("tribes", []))
        
        state = np.concatenate([
            board_features,
            unit_features,
            city_features,
            tech_tribe_features
        ])
        
        return torch.from_numpy(state).float()


class TribesModel(nn.Module):
    """
    Policy and value head model for Tribes game.
    
    Outputs:
    - action_type_logits: (batch_size, 32)
    - source_logits: (batch_size, 151)
    - target_logits: (batch_size, 163)
    - param_logits: (batch_size, 80)
    - value: (batch_size, 1)
    """
    
    def __init__(self, state_size: int = None):
        super().__init__()
        
        # If state_size not provided, use encoder's calculated size
        if state_size is None:
            encoder = StateEncoder()
            state_size = encoder.total_state_size
        
        self.state_size = state_size
        self.hidden_size = 512
        
        # Shared trunk: process state to hidden representation
        self.trunk = nn.Sequential(
            nn.Linear(state_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
        )
        
        # Policy heads: one for each action component
        self.action_type_head = nn.Linear(self.hidden_size, 32)
        self.source_head = nn.Linear(self.hidden_size, 151)
        self.target_head = nn.Linear(self.hidden_size, 163)
        self.param_head = nn.Linear(self.hidden_size, 80)
        
        # Value head for training
        self.value_head = nn.Sequential(
            nn.Linear(self.hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            state: (batch_size, state_size) tensor
        
        Returns:
            Tuple of (action_type_logits, source_logits, target_logits, param_logits, value)
        """
        hidden = self.trunk(state)
        
        action_type_logits = self.action_type_head(hidden)
        source_logits = self.source_head(hidden)
        target_logits = self.target_head(hidden)
        param_logits = self.param_head(hidden)
        value = self.value_head(hidden)
        
        return action_type_logits, source_logits, target_logits, param_logits, value
    
    def save(self, path: str) -> None:
        """Save model weights to disk."""
        torch.save(self.state_dict(), path)
    
    def load(self, path: str, device: str = "cpu") -> None:
        """Load model weights from disk."""
        self.load_state_dict(torch.load(path, map_location=device))
    
    @staticmethod
    def create_or_load(model_path: Optional[str] = None, device: str = "cpu") -> "TribesModel":
        """Factory method to create or load model."""
        model = TribesModel()
        if model_path and Path(model_path).exists():
            model.load(model_path, device=device)
        return model.to(device)


def encode_state(payload: Dict, encoder: Optional[StateEncoder] = None) -> torch.Tensor:
    """Convenience function to encode a state payload."""
    if encoder is None:
        encoder = StateEncoder()
    return encoder.encode(payload)


def load_model(model_path: str, device: str = "cpu") -> TribesModel:
    """Load a trained model from disk."""
    return TribesModel.create_or_load(model_path=model_path, device=device)


# Test/example usage
if __name__ == "__main__":
    # Get the actual state size from the encoder
    encoder = StateEncoder()
    state_size = encoder.total_state_size
    
    # Example: Create model and do a forward pass
    model = TribesModel(state_size=state_size)
    model.eval()
    
    # Dummy state tensor (batch_size=2)
    dummy_state = torch.randn(2, state_size)
    
    with torch.no_grad():
        action_type_logits, source_logits, target_logits, param_logits, value = model(dummy_state)
    
    print(f"action_type_logits shape: {action_type_logits.shape}")  # (2, 32)
    print(f"source_logits shape: {source_logits.shape}")            # (2, 151)
    print(f"target_logits shape: {target_logits.shape}")            # (2, 163)
    print(f"param_logits shape: {param_logits.shape}")              # (2, 80)
    print(f"value shape: {value.shape}")                            # (2, 1)
    
    # Test StateEncoder with JSON
    print("\n--- Testing StateEncoder ---")
    encoder = StateEncoder()
    print(f"Total state size: {encoder.total_state_size}")
    
    # Try loading examplePayload
    with open("examplePayload.json", "r") as f:
        payload = json.load(f)
    
    state_tensor = encoder.encode(payload)
    print(f"Encoded state shape: {state_tensor.shape}")
    print(f"Expected shape: ({encoder.total_state_size},)")
    
    # Forward pass with real state
    with torch.no_grad():
        batch_state = state_tensor.unsqueeze(0)  # Add batch dimension
        out = model(batch_state)
        print(f"\nForward pass with real state:")
        print(f"  action_type_logits: {out[0].shape}")
        print(f"  source_logits: {out[1].shape}")
        print(f"  target_logits: {out[2].shape}")
        print(f"  param_logits: {out[3].shape}")
        print(f"  value: {out[4].shape}")
