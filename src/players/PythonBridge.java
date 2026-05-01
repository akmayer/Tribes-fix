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

        payload.put("schema_version", 1);
        payload.put("tick", gs.getTick());
        payload.put("active_tribe_id", gs.getActiveTribeID());
        payload.put("game_mode", gs.getGameMode().name());
        payload.put("is_game_over", gs.isGameOver());
        payload.put("available_action_count", allActions.size());
        payload.put("available_actions", serializeActions(allActions));
        payload.put("board", serializeBoard(board));
        payload.put("tribes", serializeTribes(board));

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

    private static JSONArray serializeTribes(Board board) {
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
            t.put("cities", serializeCities(board, tribe));
            t.put("units", serializeTribeUnits(board, tribe));
            t.put("extra_units", serializeUnitIds(board, tribe.getExtraUnits()));
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

    private static JSONArray serializeTribeUnits(Board board, Tribe tribe) {
        JSONArray units = new JSONArray();
        Set<Integer> seen = new HashSet<>();

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

        return units;
    }

    private static JSONArray serializeCities(Board board, Tribe tribe) {
        JSONArray cities = new JSONArray();
        for (Integer cityId : tribe.getCitiesID()) {
            City city = (City) board.getActor(cityId);
            if (city == null) {
                continue;
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

    private static JSONArray serializeUnitIds(Board board, ArrayList<Integer> unitIds) {
        JSONArray units = new JSONArray();
        Set<Integer> seen = new HashSet<>();
        for (Integer unitId : unitIds) {
            if (unitId == null || seen.contains(unitId)) {
                continue;
            }
            seen.add(unitId);
            Unit unit = (Unit) board.getActor(unitId);
            if (unit != null) {
                units.put(serializeUnit(unit));
            }
        }
        return units;
    }

    private static JSONArray serializeActions(ArrayList<Action> actions) {
        JSONArray serialized = new JSONArray();
        for (int i = 0; i < actions.size(); i++) {
            Action action = actions.get(i);
            JSONObject json = new JSONObject();
            json.put("index", i);
            json.put("action_type", enumName(action.getActionType()));
            json.put("class_name", action.getClass().getSimpleName());
            json.put("description", String.valueOf(action));
            serialized.put(json);
        }
        return serialized;
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
