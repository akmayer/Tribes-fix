package players;

import core.actions.Action;
import core.game.GameState;
import org.json.JSONArray;
import org.json.JSONObject;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.Scanner;
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
                // Extract logits and masks from response
                JSONArray actionTypeLogits = policyResponse.optJSONArray("action_type_logits");
                JSONObject masks = policyResponse.optJSONObject("masks");
                JSONObject maskActionType = masks == null ? null : masks.optJSONObject("action_type_mask") != null ? null : null;

                // Sample using action_type logits + mask. If unavailable, fall back to simple masks handling.
                int actionIdx = selectActionFromPolicy(allActions, policyResponse);
                if (actionIdx >= 0 && actionIdx < allActions.size()) {
                    toExecute = allActions.get(actionIdx);
                    System.out.println("[PolicyAgent] Selected action index: " + actionIdx);
                    return toExecute;
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

    /**
     * Select an action index based on the factorized policy response.
     * Currently uses only the `action_type` logits + mask to pick an action type,
     * then selects uniformly among available actions of that type.
     */
    private int selectActionFromPolicy(ArrayList<Action> allActions, JSONObject policyResponse) {
        try {
            JSONArray logits = policyResponse.optJSONArray("action_type_logits");
            JSONObject masks = policyResponse.optJSONObject("masks");
            JSONArray actionMask = masks == null ? null : masks.optJSONArray("action_type_mask");

            if (logits == null || actionMask == null) {
                return selectActionFromMasks(allActions, masks);
            }

            double[] probs = softmaxMasked(logits, actionMask);
            int sampledType = sampleFromDistribution(probs);

            String actionTypeName = actionTypeNameFromSchema(sampledType);
            if (actionTypeName != null) {
                ArrayList<Integer> candidates = new ArrayList<>();
                for (int i = 0; i < allActions.size(); i++) {
                    Action a = allActions.get(i);
                    if (a.getActionType() != null && actionTypeName.equals(a.getActionType().name())) {
                        candidates.add(i);
                    }
                }
                if (!candidates.isEmpty()) {
                    int pick = candidates.get(rnd.nextInt(candidates.size()));
                    return pick;
                }
            }
        } catch (Exception e) {
            System.out.println("[PolicyAgent] error selecting from policy: " + e.getMessage());
        }
        // Fallback: uniform random among all actions
        if (allActions.size() > 0) return rnd.nextInt(allActions.size());
        return -1;
    }

    private int sampleFromDistribution(double[] probs) {
        double r = rnd.nextDouble();
        double c = 0.0;
        for (int i = 0; i < probs.length; i++) {
            c += probs[i];
            if (r <= c) return i;
        }
        return probs.length - 1;
    }

    private double[] softmaxMasked(JSONArray logits, JSONArray mask) {
        int n = Math.min(logits.length(), mask.length());
        double max = Double.NEGATIVE_INFINITY;
        boolean any = false;
        double[] vals = new double[n];
        for (int i = 0; i < n; i++) {
            double m = mask.optDouble(i, 0.0);
            if (m <= 0.0) {
                vals[i] = Double.NEGATIVE_INFINITY;
            } else {
                double v = logits.optDouble(i, 0.0);
                vals[i] = v;
                if (v > max) max = v;
                any = true;
            }
        }
        double[] exps = new double[n];
        double sum = 0.0;
        if (!any) {
            // No allowed entries -> uniform over everything
            for (int i = 0; i < n; i++) exps[i] = 1.0 / n;
            return exps;
        }
        for (int i = 0; i < n; i++) {
            if (vals[i] == Double.NEGATIVE_INFINITY) {
                exps[i] = 0.0;
            } else {
                double e = Math.exp(vals[i] - max);
                exps[i] = e;
                sum += e;
            }
        }
        if (sum <= 0.0) {
            // fallback uniform over allowed
            int allowed = 0;
            for (int i = 0; i < n; i++) if (exps[i] > 0) allowed++;
            double uniform = allowed > 0 ? 1.0 / allowed : 1.0 / n;
            for (int i = 0; i < n; i++) exps[i] = exps[i] > 0 ? uniform : 0.0;
            return exps;
        }
        for (int i = 0; i < n; i++) exps[i] = exps[i] / sum;
        return exps;
    }

    private static JSONObject cachedSchema = null;

    private static JSONObject loadSchema() {
        if (cachedSchema != null) return cachedSchema;
        File f = new File("py_api/action_space_schema.json");
        if (!f.exists()) return null;
        try (Scanner s = new Scanner(new FileReader(f))) {
            StringBuilder sb = new StringBuilder();
            while (s.hasNextLine()) sb.append(s.nextLine()).append('\n');
            cachedSchema = new JSONObject(sb.toString());
            return cachedSchema;
        } catch (IOException e) {
            System.out.println("[PolicyAgent] error loading schema: " + e.getMessage());
            return null;
        }
    }

    private String actionTypeNameFromSchema(int index) {
        JSONObject schema = loadSchema();
        if (schema == null) return null;
        try {
            JSONObject comps = schema.getJSONObject("components");
            JSONObject at = comps.getJSONObject("action_type");
            JSONArray values = at.getJSONArray("values");
            if (index >= 0 && index < values.length()) return values.getString(index);
        } catch (Exception e) {
            // ignore
        }
        return null;
    }

    @Override
    public Agent copy() {
        return null;
    }
}
