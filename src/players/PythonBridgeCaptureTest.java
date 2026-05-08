package players;

import core.Types;
import core.actions.Action;
import core.actions.unitactions.Move;
import core.game.GameState;
import org.json.JSONArray;
import org.json.JSONObject;
import utils.Vector2d;

import java.lang.reflect.Method;
import java.util.ArrayList;

public class PythonBridgeCaptureTest {

    static class TestAction extends Action {
        public TestAction(Types.ACTION actionType) {
            this.actionType = actionType;
        }

        @Override
        public boolean isFeasible(GameState gs) {
            return true;
        }

        @Override
        public Action copy() {
            return new TestAction(this.actionType);
        }
    }

    public static void main(String[] args) throws Exception {
        ArrayList<Action> actions = new ArrayList<>();
        actions.add(new TestAction(Types.ACTION.END_TURN));
        Move move = new Move(5);
        move.setDestination(new Vector2d(3, 5));
        actions.add(move);

        Method method = PythonBridge.class.getDeclaredMethod("serializeActions", ArrayList.class);
        method.setAccessible(true);
        JSONArray serialized = (JSONArray) method.invoke(null, actions);

        if (serialized.length() != 2) {
            System.err.println("Test failed: expected 2 serialized actions");
            System.exit(2);
        }

        JSONObject first = serialized.getJSONObject(0);
        JSONObject second = serialized.getJSONObject(1);

        if (!first.has("action_type_index") || !second.has("action_type_index")) {
            System.err.println("Test failed: missing action_type_index field");
            System.exit(3);
        }

        if (!second.has("encoded_components")) {
            System.err.println("Test failed: missing encoded_components field");
            System.exit(4);
        }

        if (first.getInt("action_type_index") != 0) {
            System.err.println("Test failed: END_TURN should map to index 0");
            System.exit(5);
        }

        if (second.getInt("action_type_index") != 1) {
            System.err.println("Test failed: MOVE should map to index 1");
            System.exit(6);
        }

        JSONObject moveComponents = second.getJSONObject("encoded_components");
        if (moveComponents.getInt("source_actor_index") != 5) {
            System.err.println("Test failed: MOVE source should encode unit id 5");
            System.exit(7);
        }

        if (moveComponents.getInt("target_actor_index") != 39) {
            System.err.println("Test failed: MOVE target should encode position (3,5) as 39");
            System.exit(8);
        }

        System.out.println("Test passed: PythonBridge serializes schema-based encoded components.");
    }
}
