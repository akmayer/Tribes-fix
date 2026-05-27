package players;

import core.Types;
import core.actions.Action;
import core.game.GameState;
import org.json.JSONArray;
import org.json.JSONObject;
import utils.ElapsedCpuTimer;

import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.ArrayList;

public class PolicySamplingTest {

    // Minimal Action implementation for testing
    static class TestAction extends Action {
        private final Types.ACTION t;
        public TestAction(Types.ACTION t) {
            this.t = t;
            this.actionType = t;
        }

        @Override
        public boolean isFeasible(GameState gs) { return true; }

        @Override
        public Action copy() { return new TestAction(this.t); }
    }

    public static void main(String[] args) throws Exception {
        BridgeAgentAttempt agent = new BridgeAgentAttempt(12345L);

        ArrayList<Action> actions = new ArrayList<>();
        // index 0 -> ATTACK, index 1 -> MOVE
        actions.add(new TestAction(Types.ACTION.ATTACK));
        actions.add(new TestAction(Types.ACTION.MOVE));

        // Build policyResponse with action_type mask allowing only MOVE (index 1)
        JSONObject policy = new JSONObject();
        JSONArray logits = new JSONArray();
        JSONArray mask = new JSONArray();
        int size = 32; // must match schema size
        for (int i = 0; i < size; i++) {
            logits.put(0.0);
            mask.put(i == 1 ? 1.0 : 0.0);
        }

        JSONObject masks = new JSONObject();
        masks.put("action_type_mask", mask);
        policy.put("action_type_logits", logits);
        policy.put("masks", masks);
        policy.put("status", "success");

        // Call private selectActionFromPolicy via reflection
        Method m = BridgeAgentAttempt.class.getDeclaredMethod("selectActionFromPolicy", ArrayList.class, GameState.class, JSONObject.class);
        m.setAccessible(true);
        int selected = (int) m.invoke(agent, actions, null, policy);

        System.out.println("Selected index: " + selected);
        if (selected < 0 || selected >= actions.size()) {
            System.err.println("Test failed: invalid selected index");
            System.exit(2);
        }
        Action a = actions.get(selected);
        String name = a.getActionType().name();
        System.out.println("Selected action type: " + name);
        if (!"MOVE".equals(name)) {
            System.err.println("Test failed: expected MOVE but got " + name);
            System.exit(3);
        }
        System.out.println("Test passed: sampled MOVE as expected.");
    }
}
