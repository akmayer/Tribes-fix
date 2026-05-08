# State Representation Analysis & Fog-of-War Compliance Report

## Executive Summary

✅ **State representation structure:** ADEQUATE (captures all necessary game state)

❌ **Fog-of-war enforcement:** BROKEN (perfect information visible to all agents)

### Recommendation
Before training the model, **MUST fix fog-of-war filtering** in `PythonBridge.java`. Training with current implementation will teach the model perfect-information strategies that don't work under fog-of-war.

---

## 1. State Structure Analysis

### 1.1 Board Representation

**Current Structure (examplePayload.json):**
```json
{
  "board": {
    "size": 11,
    "tiles": [
      [
        {
          "x": 0, "y": 0,
          "terrain": "GRASS",
          "resource": null,
          "building": null,
          "unit_id": -1,
          "unit": null
        },
        ...
      ]
    ]
  }
}
```

**Analysis:**
- ✅ Covers full 11×11 grid
- ✅ Includes terrain, resource, building on each tile
- ✅ Links to unit via `unit_id`
- ✅ Unit data embedded when present

**StateEncoder Implementation (model.py):**
```python
StateEncoder.encode_board() - line 49
├── Flattens 11x11 board to 968 features (11×11×8)
├── One-hot encodes terrain (8 types)
├── Binary encodes resource/building presence
└── Unit presence indicator
```

**Adequacy:** ✅ GOOD

---

### 1.2 Units Representation

**Current Structure:**
```json
{
  "tribes": [
    {
      "tribe_id": 0,
      "units": [
        {
          "actor_id": 1,
          "tribe_id": 0,
          "x": 5, "y": 5,
          "type": "WARRIOR",
          "status": "FRESH",
          "current_hp": 10,
          "max_hp": 10,
          "atk": 2, "def": 2, "mov": 1, "range": 1,
          "is_veteran": false,
          "has_moved": false,
          "has_attacked": false
        }
      ]
    }
  ]
}
```

**Analysis:**
- ✅ Complete unit stats
- ✅ Combat attributes (ATK, DEF, health)
- ✅ Movement info (MOV, status)
- ✅ Position coordinates
- ⚠️ **PROBLEM**: ALL units from ALL tribes included (no FOW filtering)

**StateEncoder Implementation (model.py):**
```python
StateEncoder.encode_units() - line 85
├── Limits to max 100 units
├── Extracts: type, health%, position, combat stats
├── Current limitation: only encodes active tribe's units
└── Missing: filtering for visible vs hidden enemy units
```

**Adequacy:** ⚠️ PARTIAL (structure is good, but needs FOW filtering)

---

### 1.3 Cities Representation

**Current Structure:**
```json
{
  "tribes": [
    {
      "tribe_id": 0,
      "cities": [
        {
          "actor_id": 2,
          "tribe_id": 0,
          "x": 3, "y": 4,
          "level": 1,
          "population": 5,
          "is_capital": true,
          "has_walls": false,
          "buildings": [
            {"type": "TEMPLE", "x": 3, "y": 4}
          ],
          "unit_ids": [1, 2]
        }
      ]
    }
  ]
}
```

**Analysis:**
- ✅ City development level
- ✅ Population and production capacity
- ✅ Capital/walls status
- ✅ Buildings within city
- ⚠️ **PROBLEM**: ALL cities from ALL tribes visible

**StateEncoder Implementation (model.py):**
```python
StateEncoder.encode_cities() - line 121
├── Limits to max 50 cities
├── Extracts: level, population%, position, flags
└── Current limitation: only encodes active tribe's cities
```

**Adequacy:** ⚠️ PARTIAL (needs FOW filtering)

---

### 1.4 Technology Tree

**Current Structure:**
```json
{
  "technology": {
    "researched_techs": [true, false, true, ...],
    "num_techs": 50,
    ...
  }
}
```

**Analysis:**
- ✅ Lists researched technologies
- ✅ Useful for strategy prediction
- ✅ Per-tribe included

**StateEncoder Implementation (model.py):**
```python
StateEncoder.encode_tech_and_tribe() - line 146
├── Encodes first 50 tech flags
└── Tribe stats: stars, score
```

**Adequacy:** ✅ GOOD

---

## 2. Fog-of-War Compliance Analysis

### 2.1 Current Behavior

**File: `src/players/PythonBridge.java`**

**`serializeTribes()` method (line 95):**
```java
for (Tribe tribe : board.getTribes()) {
    JSONObject t = new JSONObject();
    // ... serializes ALL tribe data
    t.put("units", serializeTribeUnits(board, tribe));    // <-- NO FOW CHECK
    t.put("cities", serializeCities(board, tribe));       // <-- NO FOW CHECK
    tribes.put(t);
}
```

**Result:** Every tribe's complete unit/city list is included in payload, regardless of active tribe's visibility.

### 2.2 The Problem

**Example Scenario:**
- Active tribe: Tribe 0
- Enemy tribe: Tribe 1
- Tribe 0 has observability grid (11×11 boolean array) tracking what it can see
- Enemy unit at position (9, 9) is OUTSIDE Tribe 0's observability grid
- **Current behavior:** Unit is still included in payload
- **Correct behavior:** Unit should NOT be in payload

### 2.3 Impact on Training

**NN Learning with Perfect Information:**
1. Model sees ALL enemy units, even those hidden from active tribe
2. Model learns to predict attacks on hidden enemies
3. Model learns positions of all enemy cities
4. Model optimizes strategy with god-like knowledge

**When Deployed with Fog-of-War:**
1. Model only sees partial information
2. Previously "learned" strategies become invalid
3. Model performs terribly because it never learned to work with partial information

### 2.4 Severity

🔴 **CRITICAL** - Training will be invalid until fixed

---

## 3. Required Fix: FOW Filtering in PythonBridge.java

### 3.1 What Needs to Change

**File:** `src/players/PythonBridge.java`

**Method 1: `serializeTribeUnits()` (line 150)**

Current:
```java
private static JSONArray serializeTribeUnits(Board board, Tribe tribe) {
    JSONArray units = new JSONArray();
    // Iterate all cities in tribe
    for (Integer cityId : tribe.getCitiesID()) {
        // Add all units from city
    }
    // Add all extra units
    for (Integer unitId : tribe.getExtraUnits()) {
        // Add unit
    }
    return units;
}
```

Should be:
```java
private static JSONArray serializeTribeUnits(Board board, Tribe tribe, boolean[][] obsGrid, int activeTribeID) {
    JSONArray units = new JSONArray();
    
    // If this is the active tribe, include all units
    if (tribe.getTribeId() == activeTribeID) {
        // ... existing code ...
    } else {
        // Enemy tribe - only include units in observability grid
        for (Integer cityId : tribe.getCitiesID()) {
            City city = (City) board.getActor(cityId);
            if (city != null) {
                for (Integer unitId : city.getUnitsID()) {
                    if (unitId != null && !seen.contains(unitId)) {
                        Unit unit = (Unit) board.getActor(unitId);
                        if (unit != null) {
                            // CHECK: Is unit in observability grid?
                            int ux = unit.getPosition().x;
                            int uy = unit.getPosition().y;
                            if (0 <= ux && ux < obsGrid.length && 
                                0 <= uy && uy < obsGrid[0].length &&
                                obsGrid[ux][uy]) {  // <-- Only if visible
                                units.put(serializeUnit(unit));
                            }
                        }
                    }
                }
            }
        }
        // Similar for extra units...
    }
    return units;
}
```

**Method 2: `serializeCities()` (line 175)**

Similar filtering needed:
```java
private static JSONArray serializeCities(Board board, Tribe tribe, boolean[][] obsGrid, int activeTribeID) {
    JSONArray cities = new JSONArray();
    for (Integer cityId : tribe.getCitiesID()) {
        City city = (City) board.getActor(cityId);
        if (city == null) continue;
        
        // Check visibility
        if (tribe.getTribeId() != activeTribeID) {
            // Enemy city - only include if visible
            int cx = city.getPosition().x;
            int cy = city.getPosition().y;
            if (!(0 <= cx && cx < obsGrid.length && 
                  0 <= cy && cy < obsGrid[0].length &&
                  obsGrid[cx][cy])) {
                continue;  // Skip hidden city
            }
        }
        
        // ... serialize city ...
    }
    return cities;
}
```

**Method 3: `buildPayload()` (line 54)**

Update to pass observability grid:
```java
private static JSONObject buildPayload(GameState gs, ArrayList<Action> allActions) {
    JSONObject payload = new JSONObject();
    Board board = gs.getBoard();
    Tribe activeTribe = gs.getActiveTribe();
    boolean[][] obsGrid = activeTribe.getObsGrid();
    int activeTribeID = gs.getActiveTribeID();
    
    // ... existing code ...
    
    payload.put("tribes", serializeTribes(board, obsGrid, activeTribeID));
    
    return payload;
}
```

### 3.2 Implementation Checklist

- [ ] Get active tribe's `obsGrid` from `gs.getActiveTribe().getObsGrid()`
- [ ] Update `buildPayload()` to extract and pass observability grid
- [ ] Update `serializeTribes()` signature to accept `obsGrid, activeTribeID`
- [ ] Add FOW check in `serializeTribeUnits()` for enemy units
- [ ] Add FOW check in `serializeCities()` for enemy cities
- [ ] Consider: Should enemy tribe metadata (stars, score) be visible? (probably yes)
- [ ] Test with game running both with and without FOW enabled
- [ ] Verify captures no longer include out-of-sight enemy units

---

## 4. Current State Size Analysis

**StateEncoder Total Size:** 3,128 features

Breakdown:
- Board (968): 11×11×8 channels
- Units (1,600): 100 units × 16 features
- Cities (500): 50 cities × 10 features
- Tech/Tribe (60): 50 + 10

**Adequacy for NN Input:** ✅ GOOD

- Large enough to capture complexity
- Small enough for efficient training
- Reasonable for 3-layer MLP with 512 hidden neurons

---

## 5. Recommendations

### Immediate (Blocking)
1. **Fix FOW in PythonBridge.java** as described in Section 3
2. **Test the fix** with game running at high speed, verify hidden units don't appear
3. **Validate captures** contain only visible information

### Short-term (Before Training)
4. **Generate diverse captures** with FOW enabled
5. **Verify state encoder handles edge cases** (0 units, 0 cities, etc.)
6. **Consider adding team fog** (if applicable) to team-shared units/cities

### Medium-term (During Training)
7. **Log state summaries** during training to catch information leaks
8. **Compare trained model performance** with/without FOW enabled
9. **Use validation set** with FOW-enabled games

### Long-term (Improvements)
10. **Consider partial observability encoding** (confidence levels for distant units)
11. **Add uncertainty features** (last-seen positions for enemies)
12. **Implement attention over visible units** (which units should we focus on?)

---

## 6. Validation Checklist

Before training starts:

- [ ] FOW filtering implemented in PythonBridge.java
- [ ] Captures generated with FOW enabled
- [ ] Sample capture inspected: no out-of-sight enemy units
- [ ] StateEncoder test pass (model.py line 330)
- [ ] app.py successfully loads model
- [ ] train.py successfully loads captures
- [ ] Model forward pass produces correct output shapes
- [ ] Fog-of-war game runs without errors

---

## Appendix: Related Code Locations

**Observability Grid:**
- Location: `src/core/actors/Tribe.java` line 47
- Type: `boolean[][] obsGrid`
- Getter: `getObsGrid()` method
- Meaning: `obsGrid[x][y] = true` → Tribe can see tile (x,y)

**Active Tribe:**
- Location: `src/core/game/GameState.java`
- Getter: `getActiveTribe()`
- Property: `getActiveTribeID()`

**Observation Grid Initialization:**
- File: `Tribe.java` line 152
- Method: `initObsGrid(int size)`
- Note: Can be all-true (no FOW) or sparse (FOW enabled)

**Key Question:** Is `obsGrid` correctly updated during gameplay?
- Check: `src/core/game/Game.java` for `updateObsGrid()` or similar
- Verify: Units only see what they can observe from their current position

