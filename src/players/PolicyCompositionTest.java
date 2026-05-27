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

public class PolicyCompositionTest {

    public static void main(String[] args) throws Exception {
        BridgeAgentAttempt agent = new BridgeAgentAttempt(12345L);

        ArrayList<Action> actions = new ArrayList<>();

        Move moveA = new Move(5);
        moveA.setDestination(new Vector2d(3, 5));
        actions.add(moveA);

        Move moveB = new Move(5);
        moveB.setDestination(new Vector2d(3, 6));
        actions.add(moveB);

        actions.add(new core.actions.tribeactions.EndTurn(0));

        JSONObject policy = new JSONObject();
        policy.put("status", "success");

        JSONArray actionTypeProbs = zeros(32);
        actionTypeProbs.put(1, 1.0); // MOVE
        policy.put("action_type_probs", actionTypeProbs);

        JSONArray sourceProbs = zeros(151);
        sourceProbs.put(5, 1.0);
        policy.put("source_probs", sourceProbs);

        JSONArray targetProbs = zeros(163);
        targetProbs.put(encoderTargetIndex(3, 5), 1.0);
        policy.put("target_probs", targetProbs);

        JSONArray paramProbs = zeros(80);
        paramProbs.put(0, 1.0);
        policy.put("param_probs", paramProbs);

        Method m = BridgeAgentAttempt.class.getDeclaredMethod("selectActionFromPolicy", ArrayList.class, GameState.class, JSONObject.class);
        m.setAccessible(true);
        int selected = (int) m.invoke(agent, actions, null, policy);

        if (selected != 0) {
            System.err.println("Test failed: expected the first MOVE action to win the joint probability, got index " + selected);
            System.exit(2);
        }

        System.out.println("Test passed: full factorized probability composition selected the expected action.");
    }

    private static JSONArray zeros(int size) {
        JSONArray arr = new JSONArray();
        for (int i = 0; i < size; i++) {
            arr.put(0.0);
        }
        return arr;
    }

    private static int encoderTargetIndex(int x, int y) {
        return x * 11 + y + 1;
    }
}
