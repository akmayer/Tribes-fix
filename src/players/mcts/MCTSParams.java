package players.mcts;

import players.heuristics.AlgParams;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

@SuppressWarnings("WeakerAccess")
public class MCTSParams extends AlgParams {

    // Parameters
    public double K = Math.sqrt(2);
    public int ROLLOUT_LENGTH = 10;//10;
    public boolean ROLOUTS_ENABLED = true;
    public boolean CAPTURE_MCTS = false;

    // AlphaZero-style neural MCTS (PUCT)
    public boolean NEURAL_PRIORS = false;
    public boolean NEURAL_VALUE = false;
    public double CPUCT = 1.5;

    public boolean USE_UNIFORM_PRIORS = true;
    public double UNIFORM_PRIOR_WEIGHT = 1.0;

    public boolean DIRICHLET_ROOT_NOISE = true;
    public double DIRICHLET_ALPHA = 0.3;
    public double DIRICHLET_EPSILON = 0.25;

    public void setParameterValue(String param, Object value) {
        switch(param) {
            case "K": K = (double) value; break;
            case "ROLLOUT_LENGTH": ROLLOUT_LENGTH = (int) value; break;
            case "heuristic_method": heuristic_method = (int) value; break;
            case "NEURAL_PRIORS": NEURAL_PRIORS = (boolean) value; break;
            case "NEURAL_VALUE": NEURAL_VALUE = (boolean) value; break;
            case "CPUCT": CPUCT = (double) value; break;
        }
    }

    public Object getParameterValue(String param) {
        switch(param) {
            case "K": return K;
            case "ROLLOUT_LENGTH": return ROLLOUT_LENGTH;
            case "heuristic_method": return heuristic_method;
            case "NEURAL_PRIORS": return NEURAL_PRIORS;
            case "NEURAL_VALUE": return NEURAL_VALUE;
            case "CPUCT": return CPUCT;
        }
        return null;
    }


    public ArrayList<String> getParameters() {
        ArrayList<String> paramList = new ArrayList<>();
        paramList.add("K");
        paramList.add("rollout_depth");
        paramList.add("heuristic_method");
        paramList.add("NEURAL_PRIORS");
        paramList.add("NEURAL_VALUE");
        paramList.add("CPUCT");
        return paramList;
    }

    public Map<String, Object[]> getParameterValues() {
        HashMap<String, Object[]> parameterValues = new HashMap<>();
        parameterValues.put("K", new Double[]{1.0, Math.sqrt(2), 2.0});
        parameterValues.put("rollout_depth", new Integer[]{5, 8, 10, 12, 15});
        parameterValues.put("heuristic_method", new Integer[]{ENTROPY_HEURISTIC, SIMPLE_HEURISTIC, DIFF_HEURISTIC});
        parameterValues.put("NEURAL_PRIORS", new Boolean[]{false, true});
        parameterValues.put("NEURAL_VALUE", new Boolean[]{false, true});
        parameterValues.put("CPUCT", new Double[]{0.5, 1.0, 1.5, 2.0});
        return parameterValues;
    }

}
