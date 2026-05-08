package players;

import core.actions.Action;
import core.game.GameState;
import org.json.JSONArray;
import org.json.JSONObject;
import utils.ElapsedCpuTimer;

import java.util.ArrayList;
import java.util.Random;

public class RandomAgent extends Agent {

    private Random rnd;

    public RandomAgent(long seed)
    {
        super(seed);
        rnd = new Random(seed);
    }

    @Override
    public Action act(GameState gs, ElapsedCpuTimer ect)
    {
        ArrayList<Action> allActions = gs.getAllAvailableActions();
        Action toExecute = null;
        
        try {
            String resp = PythonBridge.queryPolicy(gs, allActions);
            System.out.println("[PythonBridge response] " + resp);
            
            // Parse the policy response
            JSONObject policyResponse = new JSONObject(resp);
            String status = policyResponse.optString("status", "error");
            
            if ("success".equals(status)) {
                // Extract masks from response
                JSONObject masks = policyResponse.optJSONObject("masks");
                if (masks != null) {
                    // For now, use the masks to filter available actions
                    // In practice, you'd use the logits to sample probabilistically
                    int actionIdx = selectActionFromMasks(allActions, masks);
                    if (actionIdx >= 0 && actionIdx < allActions.size()) {
                        toExecute = allActions.get(actionIdx);
                        System.out.println("[PolicyAgent] Selected action index: " + actionIdx);
                        return toExecute;
                    }
                }
            }
        } catch (Exception e) {
            // If the bridge is not available, fall back to random action
            System.out.println("[PythonBridge] error: " + e.getMessage() + ". Falling back to random.");
        }

        // Fallback: select random action
        int nActions = allActions.size();
        if (nActions > 0) {
            int selectedIdx = rnd.nextInt(nActions);
            toExecute = allActions.get(selectedIdx);
            System.out.println("[RandomAgent] Selected random action index: " + selectedIdx);
        }
        
        return toExecute;
    }
    
    /**
     * Select an action based on the masks returned from the policy.
     * This is a simple greedy selection: pick the first legal action in the available list.
     */
    private int selectActionFromMasks(ArrayList<Action> allActions, JSONObject masks) {
        // For now, just return the first available action (they're all masked)
        // In a more sophisticated version, you'd:
        // 1. Convert each available action to component indices
        // 2. Check if the component combination is legal given the masks
        // 3. Sample probabilistically or greedily
        
        if (allActions.size() > 0) {
            return 0; // Return first action as placeholder
        }
        return -1;
    }

    @Override
    public Agent copy() {
        return null;
    }
}
