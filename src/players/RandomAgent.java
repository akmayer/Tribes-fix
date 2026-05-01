package players;

import core.actions.Action;
import core.game.GameState;
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
        int nActions = allActions.size();
        Action toExecute = null;
        try {
            String json = String.format("{\"tick\": %d, \"n_actions\": %d}", gs.getTick(), nActions);
            String resp = PythonBridge.queryPolicy(json);
            System.out.println("[PythonBridge response] " + resp);
        } catch (Exception e) {
            // If the bridge is not available, fall back to random action
            System.out.println("[PythonBridge] error: " + e.getMessage() + ". Falling back to random.");
        }

        if (nActions > 0) {
            toExecute = allActions.get(rnd.nextInt(nActions));
        }
        return toExecute;
    }

    @Override
    public Agent copy() {
        return null;
    }
}
