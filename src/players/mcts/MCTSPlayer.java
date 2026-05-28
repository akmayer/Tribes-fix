package players.mcts;

import core.actions.Action;
import core.actions.tribeactions.EndTurn;
import core.actions.tribeactions.SendStars;
import core.actors.Tribe;
import core.game.Game;
import core.game.GameState;
import core.Types;
import core.Constants;
import players.Agent;
import players.PythonBridge;
import utils.ElapsedCpuTimer;

import java.util.ArrayList;
import java.util.IdentityHashMap;
import java.util.Random;

import static core.Constants.TURN_TIME_MILLIS;

public class MCTSPlayer extends Agent {

    private Random m_rnd;
    private MCTSParams params;

    public MCTSPlayer(long seed)
    {
        super(seed);
        m_rnd = new Random(seed);
        this.params = new MCTSParams();
    }

    public MCTSPlayer(long seed, MCTSParams params) {
        this(seed);
        this.params = params;
    }

    public Action act(GameState gs, ElapsedCpuTimer ect) {
        //Gather all available actions:
        ArrayList<Action> allActions = gs.getAllAvailableActions();

        if(allActions.size() == 1)
            return allActions.get(0); //EndTurn, it's possible.

        ArrayList<Action> rootActions = params.PRIORITIZE_ROOT ? determineActionGroup(gs, m_rnd) : allActions;
        if(rootActions == null)
            return new EndTurn();

        SingleTreeNode m_root = new SingleTreeNode(params, m_rnd, rootActions.size(), rootActions, this.playerID);
        m_root.setRootGameState(m_root, gs, allPlayerIDs);

        m_root.mctsSearch(ect);

        if (params.CAPTURE_MCTS) {
            int[] rootVisits = m_root.getVisitCounts();
            int[] visitCounts = alignVisitCounts(allActions, rootActions, rootVisits);
            PythonBridge.captureMctsSample(gs, allActions, visitCounts, m_root.getRootValue(), playerID);
        }
        Action chosen = rootActions.get(m_root.mostVisitedAction());

        if (true) {
            System.out.println("\n===== MCTS DECISION =====");
            System.out.println("Player: " + playerID);
            System.out.println("Available actions: " + rootActions.size());
            System.out.println("Chosen action: " + chosen);

            int[] rootVisits = m_root.getVisitCounts();
            int[] visitCounts = alignVisitCounts(allActions, rootActions, rootVisits);

            System.out.println("---- Visit counts ----");
            for (int i = 0; i < allActions.size(); i++) {
                Action a = allActions.get(i);
                int v = visitCounts[i];

                // highlight chosen action
                boolean isChosen = a == chosen;

                System.out.println(
                    (isChosen ? ">> " : "   ") +
                    "[" + i + "] " + a +
                    " | visits=" + v +
                    actionDeltaSummary(gs, a)
                );
            }

            System.out.println("========================\n");
        }

        return chosen;


    }

    private String actionDeltaSummary(GameState gs, Action action) {
        try {
            int activeTribeID = gs.getActiveTribeID();
            Tribe beforeActive = gs.getTribe(activeTribeID);
            int beforeActiveScore = beforeActive.getScore();
            int beforeActiveStars = beforeActive.getStars();

            Integer targetTribeID = null;
            int beforeTargetScore = 0;
            int beforeTargetStars = 0;
            if (action instanceof SendStars) {
                targetTribeID = ((SendStars) action).getTargetID();
                Tribe beforeTarget = gs.getTribe(targetTribeID);
                beforeTargetScore = beforeTarget.getScore();
                beforeTargetStars = beforeTarget.getStars();
            }

            GameState next = gs.copy();
            next.advance(action, false);

            Tribe afterActive = next.getTribe(activeTribeID);
            String summary = " | p" + activeTribeID
                    + " score " + beforeActiveScore + "->" + afterActive.getScore()
                    + ", stars " + beforeActiveStars + "->" + afterActive.getStars();

            if (targetTribeID != null) {
                Tribe afterTarget = next.getTribe(targetTribeID);
                summary += " | target p" + targetTribeID
                        + " score " + beforeTargetScore + "->" + afterTarget.getScore()
                        + ", stars " + beforeTargetStars + "->" + afterTarget.getStars();
            }

            return summary;
        } catch (Exception e) {
            return " | delta unavailable: " + e.getMessage();
        }
    }


    @Override
    public Agent copy() {
        return null;
    }

    @Override
    public void result(GameState gs, double reward) {
        if (!params.CAPTURE_MCTS) {
            return;
        }
        Types.RESULT winner = Types.RESULT.INCOMPLETE;
        try {
            winner = gs.getTribe(playerID).getWinner();
        } catch (Exception ignored) {
            // keep default
        }
        PythonBridge.reportGameResult(gs, playerID, reward, winner);
    }

    private int[] alignVisitCounts(ArrayList<Action> allActions, ArrayList<Action> rootActions, int[] rootVisits) {
        int[] aligned = new int[allActions.size()];
        if (rootActions == allActions) {
            System.arraycopy(rootVisits, 0, aligned, 0, Math.min(rootVisits.length, aligned.length));
            return aligned;
        }

        IdentityHashMap<Action, Integer> indexByAction = new IdentityHashMap<>();
        for (int i = 0; i < allActions.size(); i++) {
            indexByAction.put(allActions.get(i), i);
        }

        int limit = Math.min(rootActions.size(), rootVisits.length);
        for (int i = 0; i < limit; i++) {
            Integer idx = indexByAction.get(rootActions.get(i));
            if (idx != null) {
                aligned[idx] = rootVisits[i];
            }
        }

        return aligned;
    }

}
