package players;

import core.TechnologyTree;
import core.Types;
import core.actions.Action;
import core.actors.Building;
import core.actors.City;
import core.actors.Tribe;
import core.actors.units.Unit;
import core.game.Board;
import core.game.GameState;
import org.json.JSONArray;
import org.json.JSONObject;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.Set;

public class PythonBridge {

    private static final String DEFAULT_URL = "http://127.0.0.1:8000/query";
    private static JSONObject cachedSchema = null;

    public static String queryPolicy(GameState gs, ArrayList<Action> allActions) throws IOException {
        JSONObject payload = buildPayload(gs, allActions);
        return postJson(payload.toString());
    }

    private static String postJson(String jsonPayload) throws IOException {
        URL url = new URL(DEFAULT_URL);
        HttpURLConnection conn = (HttpURLConnection) url.openConnection();
        conn.setRequestMethod("POST");
        conn.setRequestProperty("Content-Type", "application/json; utf-8");
        conn.setRequestProperty("Accept", "application/json");
        conn.setDoOutput(true);

        try (OutputStream os = conn.getOutputStream()) {
            byte[] input = jsonPayload.getBytes("utf-8");
            os.write(input, 0, input.length);
        }

        int code = conn.getResponseCode();
        BufferedReader br;
        if (code >= 200 && code < 300) {
            br = new BufferedReader(new InputStreamReader(conn.getInputStream(), "utf-8"));
        } else {
            br = new BufferedReader(new InputStreamReader(conn.getErrorStream(), "utf-8"));
        }

        StringBuilder resp = new StringBuilder();
        String responseLine = null;
        while ((responseLine = br.readLine()) != null) {
            resp.append(responseLine.trim());
        }
        return resp.toString();
    }

    private static JSONObject buildPayload(GameState gs, ArrayList<Action> allActions) {
        JSONObject payload = new JSONObject();
        Board board = gs.getBoard();
        
        // Extract observability grid for FOW filtering
        Tribe activeTribe = gs.getActiveTribe();
        boolean[][] obsGrid = activeTribe != null ? activeTribe.getObsGrid() : null;
        int activeTribeID = gs.getActiveTribeID();

        payload.put("schema_version", 1);
        payload.put("tick", gs.getTick());
        payload.put("active_tribe_id", activeTribeID);
        payload.put("game_mode", gs.getGameMode().name());
        payload.put("is_game_over", gs.isGameOver());
        payload.put("available_action_count", allActions.size());
        payload.put("available_actions", serializeActions(allActions, gs));
        payload.put("board", serializeBoard(board));
        payload.put("tribes", serializeTribes(board, obsGrid, activeTribeID));

        return payload;
    }

    private static JSONObject serializeBoard(Board board) {
        JSONObject json = new JSONObject();
        int size = board.getSize();

        json.put("size", size);
        json.put("active_tribe_id", board.getActiveTribeID());
        json.put("actor_id_counter", board.getActorIDcounter());
        json.put("capital_ids", toIntArray(board.getCapitalIDs()));

        JSONArray tiles = new JSONArray();
        for (int x = 0; x < size; x++) {
            JSONArray row = new JSONArray();
            for (int y = 0; y < size; y++) {
                JSONObject tile = new JSONObject();
                tile.put("x", x);
                tile.put("y", y);
                tile.put("terrain", enumName(board.getTerrainAt(x, y)));
                tile.put("resource", enumName(board.getResourceAt(x, y)));
                tile.put("building", enumName(board.getBuildingAt(x, y)));
                tile.put("city_id", board.getCityIdAt(x, y));
                int unitId = board.getUnits()[x][y];
                tile.put("unit_id", unitId);
                if (unitId != -1) {
                    tile.put("unit", serializeUnit((Unit) board.getActor(unitId)));
                }
                row.put(tile);
            }
            tiles.put(row);
        }
        json.put("tiles", tiles);

        return json;
    }

    private static JSONArray serializeTribes(Board board, boolean[][] obsGrid, int activeTribeID) {
        JSONArray tribes = new JSONArray();
        for (Tribe tribe : board.getTribes()) {
            JSONObject t = new JSONObject();
            t.put("tribe_id", tribe.getTribeId());
            t.put("type", enumName(tribe.getType()));
            t.put("name", tribe.getName());
            t.put("stars", tribe.getStars());
            t.put("score", tribe.getScore());
            t.put("winner", enumName(tribe.getWinner()));
            t.put("capital_id", tribe.getCapitalID());
            t.put("cities", serializeCities(board, tribe, obsGrid, activeTribeID));
            t.put("units", serializeTribeUnits(board, tribe, obsGrid, activeTribeID));
            t.put("extra_units", serializeUnitIds(board, tribe.getExtraUnits(), obsGrid, activeTribeID, tribe.getTribeId()));
            t.put("connected_cities", toIntArray(tribe.getConnectedCities()));
            t.put("tribes_met", toIntArray(tribe.getTribesMet()));
            t.put("n_kills", tribe.getnKills());
            t.put("n_pacifist_count", tribe.getnPacifistCount());
            t.put("stars_sent", tribe.getStarsSent());
            t.put("has_declared_war", tribe.getHasDeclaredWar());
            t.put("n_wars_declared", tribe.getnWarsDeclared());
            t.put("n_stars_sent", tribe.getnStarsSent());
            t.put("technology", serializeTechnologyTree(tribe.getTechTree()));
            t.put("monuments", serializeMonuments(tribe));
            tribes.put(t);
        }
        return tribes;
    }

    private static JSONArray serializeTribeUnits(Board board, Tribe tribe, boolean[][] obsGrid, int activeTribeID) {
        JSONArray units = new JSONArray();
        Set<Integer> seen = new HashSet<>();

        // If this is the active tribe, include all units (no FOW for self)
        if (tribe.getTribeId() == activeTribeID) {
            for (Integer cityId : tribe.getCitiesID()) {
                City city = (City) board.getActor(cityId);
                if (city == null) {
                    continue;
                }
                for (Integer unitId : city.getUnitsID()) {
                    if (unitId == null || seen.contains(unitId)) {
                        continue;
                    }
                    seen.add(unitId);
                    Unit unit = (Unit) board.getActor(unitId);
                    if (unit != null) {
                        units.put(serializeUnit(unit));
                    }
                }
            }

            for (Integer unitId : tribe.getExtraUnits()) {
                if (unitId == null || seen.contains(unitId)) {
                    continue;
                }
                seen.add(unitId);
                Unit unit = (Unit) board.getActor(unitId);
                if (unit != null) {
                    units.put(serializeUnit(unit));
                }
            }
        } else {
            // For enemy tribes, only include units in observability grid (FOW)
            for (Integer cityId : tribe.getCitiesID()) {
                City city = (City) board.getActor(cityId);
                if (city == null) {
                    continue;
                }
                for (Integer unitId : city.getUnitsID()) {
                    if (unitId == null || seen.contains(unitId)) {
                        continue;
                    }
                    seen.add(unitId);
                    Unit unit = (Unit) board.getActor(unitId);
                    if (unit != null && isPositionVisible(obsGrid, unit.getPosition().x, unit.getPosition().y)) {
                        units.put(serializeUnit(unit));
                    }
                }
            }

            for (Integer unitId : tribe.getExtraUnits()) {
                if (unitId == null || seen.contains(unitId)) {
                    continue;
                }
                seen.add(unitId);
                Unit unit = (Unit) board.getActor(unitId);
                if (unit != null && isPositionVisible(obsGrid, unit.getPosition().x, unit.getPosition().y)) {
                    units.put(serializeUnit(unit));
                }
            }
        }

        return units;
    }

    private static JSONArray serializeCities(Board board, Tribe tribe, boolean[][] obsGrid, int activeTribeID) {
        JSONArray cities = new JSONArray();
        for (Integer cityId : tribe.getCitiesID()) {
            City city = (City) board.getActor(cityId);
            if (city == null) {
                continue;
            }
            
            // FOW CHECK: Only include city if visible (or if it's our tribe)
            if (tribe.getTribeId() != activeTribeID) {
                // Enemy city - check visibility
                if (!isPositionVisible(obsGrid, city.getPosition().x, city.getPosition().y)) {
                    continue;  // Skip this hidden city
                }
            }
            
            JSONObject c = new JSONObject();
            c.put("actor_id", city.getActorId());
            c.put("tribe_id", city.getTribeId());
            c.put("x", city.getPosition().x);
            c.put("y", city.getPosition().y);
            c.put("level", city.getLevel());
            c.put("population", city.getPopulation());
            c.put("population_need", city.getPopulation_need());
            c.put("is_capital", city.isCapital());
            c.put("production", city.getProduction());
            c.put("has_walls", city.hasWalls());
            c.put("bound", city.getBound());
            c.put("points_worth", city.getPointsWorth());
            c.put("unit_ids", toIntArray(city.getUnitsID()));
            c.put("buildings", serializeBuildings(city));
            cities.put(c);
        }
        return cities;
    }

    private static JSONArray serializeBuildings(City city) {
        JSONArray buildings = new JSONArray();
        for (Building building : city.getBuildings()) {
            JSONObject b = new JSONObject();
            b.put("x", building.position.x);
            b.put("y", building.position.y);
            b.put("type", enumName(building.type));
            b.put("city_id", building.cityId);
            if (building instanceof core.actors.Temple) {
                b.put("kind", "TEMPLE");
            } else {
                b.put("kind", "BUILDING");
            }
            buildings.put(b);
        }
        return buildings;
    }

    private static JSONObject serializeUnit(Unit unit) {
        if (unit == null) {
            return null;
        }

        JSONObject json = new JSONObject();
        json.put("actor_id", unit.getActorId());
        json.put("tribe_id", unit.getTribeId());
        json.put("city_id", unit.getCityId());
        json.put("x", unit.getPosition().x);
        json.put("y", unit.getPosition().y);
        json.put("type", enumName(unit.getType()));
        json.put("status", enumName(unit.getStatus()));
        json.put("current_hp", unit.getCurrentHP());
        json.put("max_hp", unit.getMaxHP());
        json.put("kills", unit.getKills());
        json.put("is_veteran", unit.isVeteran());
        json.put("atk", unit.ATK);
        json.put("def", unit.DEF);
        json.put("mov", unit.MOV);
        json.put("range", unit.RANGE);
        json.put("cost", unit.COST);
        json.put("has_moved", unit.getStatus() == Types.TURN_STATUS.MOVED || unit.getStatus() == Types.TURN_STATUS.MOVED_AND_ATTACKED);
        json.put("has_attacked", unit.getStatus() == Types.TURN_STATUS.ATTACKED || unit.getStatus() == Types.TURN_STATUS.MOVED_AND_ATTACKED);
        return json;
    }

    private static JSONArray serializeUnitIds(Board board, ArrayList<Integer> unitIds, boolean[][] obsGrid, int activeTribeID, int tribeId) {
        JSONArray units = new JSONArray();
        Set<Integer> seen = new HashSet<>();
        for (Integer unitId : unitIds) {
            if (unitId == null || seen.contains(unitId)) {
                continue;
            }
            seen.add(unitId);
            Unit unit = (Unit) board.getActor(unitId);
            if (unit != null) {
                // Check visibility for enemy units
                if (tribeId != activeTribeID) {
                    if (!isPositionVisible(obsGrid, unit.getPosition().x, unit.getPosition().y)) {
                        continue;
                    }
                }
                units.put(serializeUnit(unit));
            }
        }
        return units;
    }

    private static boolean isPositionVisible(boolean[][] obsGrid, int x, int y) {
        /**
         * Check if a position is visible in the observability grid.
         * Returns true if:
         * 1. obsGrid is null (no FOW restriction)
         * 2. Position is within grid bounds and marked as observable
         */
        if (obsGrid == null) {
            return true;
        }
        if (x < 0 || y < 0 || x >= obsGrid.length) {
            return false;
        }
        if (y >= obsGrid[0].length) {
            return false;
        }
        return obsGrid[x][y];
    }

    public static JSONObject encodeActionComponents(Action action, GameState gs) {
        JSONObject components = new JSONObject();
        int actionTypeIndex = actionTypeIndexFromSchema(action.getActionType());
        components.put("action_type_index", actionTypeIndex);
        components.put("source_actor_index", sourceActorIndex(action));
        components.put("target_actor_index", targetActorIndex(action, gs));
        components.put("param_index", paramIndex(action));
        return components;
    }

    private static JSONArray serializeActions(ArrayList<Action> actions) {
        return serializeActions(actions, null);
    }

    private static JSONArray serializeActions(ArrayList<Action> actions, GameState gs) {
        JSONArray serialized = new JSONArray();
        for (int i = 0; i < actions.size(); i++) {
            Action action = actions.get(i);
            JSONObject json = new JSONObject();
            json.put("index", i);
            json.put("action_type", enumName(action.getActionType()));
            json.put("action_type_index", actionTypeIndexFromSchema(action.getActionType()));
            json.put("encoded_components", encodeActionComponents(action, gs));
            json.put("class_name", action.getClass().getSimpleName());
            json.put("description", String.valueOf(action));
            serialized.put(json);
        }
        return serialized;
    }

    private static int sourceActorIndex(Action action) {
        if (action instanceof core.actions.unitactions.UnitAction) {
            return ((core.actions.unitactions.UnitAction) action).getUnitId();
        }
        if (action instanceof core.actions.cityactions.CityAction) {
            return ((core.actions.cityactions.CityAction) action).getCityId() + 100;
        }
        return 0;
    }

    private static int targetActorIndex(Action action, GameState gs) {
        if (action instanceof core.actions.unitactions.Move) {
            utils.Vector2d destination = ((core.actions.unitactions.Move) action).getDestination();
            return destination == null ? 0 : positionToTargetIndex(destination.x, destination.y);
        }
        if (action instanceof core.actions.unitactions.Attack) {
            return ((core.actions.unitactions.Attack) action).getTargetId() + 121;
        }
        if (action instanceof core.actions.unitactions.Convert) {
            return ((core.actions.unitactions.Convert) action).getTargetId() + 121;
        }
        if (action instanceof core.actions.unitactions.Capture) {
            core.actions.unitactions.Capture capture = (core.actions.unitactions.Capture) action;
            if (capture.getCaptureType() == core.Types.TERRAIN.CITY) {
                return capture.getTargetCity() + 221;
            }
            if (capture.getCaptureType() == core.Types.TERRAIN.VILLAGE) {
                if (gs != null) {
                    core.actors.units.Unit unit = (core.actors.units.Unit) gs.getActor(capture.getUnitId());
                    if (unit != null && unit.getPosition() != null) {
                        return positionToTargetIndex(unit.getPosition().x, unit.getPosition().y);
                    }
                }
            }
            return 0;
        }
        if (action instanceof core.actions.tribeactions.BuildRoad) {
            utils.Vector2d position = ((core.actions.tribeactions.BuildRoad) action).getPosition();
            return position == null ? 0 : positionToTargetIndex(position.x, position.y);
        }
        if (action instanceof core.actions.tribeactions.SendStars) {
            return ((core.actions.tribeactions.SendStars) action).getTargetID() + 271;
        }
        if (action instanceof core.actions.tribeactions.DeclareWar) {
            return ((core.actions.tribeactions.DeclareWar) action).getTargetID() + 271;
        }
        if (action instanceof core.actions.cityactions.Build) {
            utils.Vector2d targetPos = ((core.actions.cityactions.Build) action).getTargetPos();
            return targetPos == null ? 0 : positionToTargetIndex(targetPos.x, targetPos.y);
        }
        if (action instanceof core.actions.cityactions.BurnForest) {
            utils.Vector2d targetPos = ((core.actions.cityactions.BurnForest) action).getTargetPos();
            return targetPos == null ? 0 : positionToTargetIndex(targetPos.x, targetPos.y);
        }
        if (action instanceof core.actions.cityactions.ClearForest) {
            utils.Vector2d targetPos = ((core.actions.cityactions.ClearForest) action).getTargetPos();
            return targetPos == null ? 0 : positionToTargetIndex(targetPos.x, targetPos.y);
        }
        if (action instanceof core.actions.cityactions.Destroy) {
            utils.Vector2d targetPos = ((core.actions.cityactions.Destroy) action).getTargetPos();
            return targetPos == null ? 0 : positionToTargetIndex(targetPos.x, targetPos.y);
        }
        if (action instanceof core.actions.cityactions.GrowForest) {
            utils.Vector2d targetPos = ((core.actions.cityactions.GrowForest) action).getTargetPos();
            return targetPos == null ? 0 : positionToTargetIndex(targetPos.x, targetPos.y);
        }
        if (action instanceof core.actions.cityactions.LevelUp) {
            utils.Vector2d targetPos = ((core.actions.cityactions.LevelUp) action).getTargetPos();
            return targetPos == null ? 0 : positionToTargetIndex(targetPos.x, targetPos.y);
        }
        if (action instanceof core.actions.cityactions.ResourceGathering) {
            utils.Vector2d targetPos = ((core.actions.cityactions.ResourceGathering) action).getTargetPos();
            return targetPos == null ? 0 : positionToTargetIndex(targetPos.x, targetPos.y);
        }
        if (action instanceof core.actions.unitactions.Examine) {
            core.actions.unitactions.UnitAction unitAction = (core.actions.unitactions.UnitAction) action;
            if (gs != null) {
                core.actors.units.Unit unit = (core.actors.units.Unit) gs.getActor(unitAction.getUnitId());
                if (unit != null && unit.getPosition() != null) {
                    return positionToTargetIndex(unit.getPosition().x, unit.getPosition().y);
                }
            }
            return 0;
        }
        return 0;
    }

    private static int paramIndex(Action action) {
        if (action instanceof core.actions.tribeactions.SendStars) {
            return ((core.actions.tribeactions.SendStars) action).getNumStars();
        }
        if (action instanceof core.actions.tribeactions.ResearchTech) {
            Types.TECHNOLOGY tech = ((core.actions.tribeactions.ResearchTech) action).getTech();
            return tech == null ? 0 : tech.ordinal();
        }
        if (action instanceof core.actions.cityactions.Build) {
            Types.BUILDING buildingType = ((core.actions.cityactions.Build) action).getBuildingType();
            return buildingType == null ? 0 : buildingType.ordinal();
        }
        if (action instanceof core.actions.cityactions.Spawn) {
            Types.UNIT unitType = ((core.actions.cityactions.Spawn) action).getUnitType();
            return unitType == null ? 0 : unitType.ordinal();
        }
        if (action instanceof core.actions.cityactions.LevelUp) {
            Types.CITY_LEVEL_UP bonus = ((core.actions.cityactions.LevelUp) action).getBonus();
            return bonus == null ? 0 : bonus.ordinal();
        }
        if (action instanceof core.actions.cityactions.ResourceGathering) {
            Types.RESOURCE resource = ((core.actions.cityactions.ResourceGathering) action).getResource();
            return resource == null ? 0 : resource.ordinal();
        }
        if (action instanceof core.actions.unitactions.Examine) {
            Types.EXAMINE_BONUS bonus = ((core.actions.unitactions.Examine) action).getBonus();
            return bonus == null ? 0 : bonus.ordinal();
        }
        return 0;
    }

    private static int positionToTargetIndex(int x, int y) {
        int boardSize = 11;
        JSONObject schema = loadSchema();
        if (schema != null) {
            boardSize = schema.optInt("board_size", boardSize);
        }
        return x * boardSize + y + 1;
    }

    private static JSONObject loadSchema() {
        if (cachedSchema != null) {
            return cachedSchema;
        }

        java.io.File schemaFile = new java.io.File("py_api/action_space_schema.json");
        if (!schemaFile.exists()) {
            return null;
        }

        try (java.io.FileReader reader = new java.io.FileReader(schemaFile)) {
            StringBuilder sb = new StringBuilder();
            char[] buffer = new char[4096];
            int read;
            while ((read = reader.read(buffer)) != -1) {
                sb.append(buffer, 0, read);
            }
            cachedSchema = new JSONObject(sb.toString());
            return cachedSchema;
        } catch (IOException e) {
            System.out.println("[PythonBridge] error loading schema: " + e.getMessage());
            return null;
        }
    }

    private static Integer actionTypeIndexFromSchema(Types.ACTION actionType) {
        if (actionType == null) {
            return null;
        }

        JSONObject schema = loadSchema();
        if (schema == null) {
            return null;
        }

        try {
            JSONObject components = schema.getJSONObject("components");
            JSONObject actionTypeComponent = components.getJSONObject("action_type");
            JSONObject indexMap = actionTypeComponent.getJSONObject("index_map");
            String key = actionType.name();
            if (indexMap.has(key)) {
                return indexMap.getInt(key);
            }
        } catch (Exception e) {
            System.out.println("[PythonBridge] error reading action type index: " + e.getMessage());
        }

        return null;
    }

    private static JSONObject serializeTechnologyTree(TechnologyTree techTree) {
        JSONObject json = new JSONObject();
        boolean[] researched = techTree.getResearched();
        JSONArray researchedTechs = new JSONArray();
        JSONArray researchedFlags = new JSONArray();

        for (Types.TECHNOLOGY technology : Types.TECHNOLOGY.values()) {
            boolean isResearched = researched[technology.ordinal()];
            researchedFlags.put(isResearched);
            if (isResearched) {
                researchedTechs.put(technology.name());
            }
        }

        json.put("researched", researchedTechs);
        json.put("researched_flags", researchedFlags);
        json.put("everything_researched", techTree.isEverythingResearched());
        json.put("num_researched", techTree.getNumResearched());
        return json;
    }

    private static JSONObject serializeMonuments(Tribe tribe) {
        JSONObject json = new JSONObject();
        for (Types.BUILDING building : tribe.getMonuments().keySet()) {
            json.put(building.name(), tribe.getMonuments().get(building).name());
        }
        return json;
    }

    private static String enumName(Enum<?> value) {
        return value == null ? null : value.name();
    }

    private static JSONArray toIntArray(int[] values) {
        JSONArray array = new JSONArray();
        if (values == null) {
            return array;
        }
        for (int value : values) {
            array.put(value);
        }
        return array;
    }

    private static JSONArray toIntArray(ArrayList<Integer> values) {
        JSONArray array = new JSONArray();
        if (values == null) {
            return array;
        }
        for (Integer value : values) {
            array.put(value);
        }
        return array;
    }

}
