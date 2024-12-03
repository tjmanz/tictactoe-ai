package ticTacToe;

import java.util.HashMap;
import java.util.List;
import java.util.Random;

/**
 * A Tic-Tac-Toe agent that learns the best strategy using Policy Iteration.
 * The agent alternates between evaluating its current policy and improving it
 * until it converges to the optimal policy.
 */
public class PolicyIterationAgent extends Agent {

    /**
     * This map is used to store the values of states according to the current policy (policy evaluation). 
     */
    HashMap<Game, Double> policyValues = new HashMap<>();

    /**
     * This stores the current policy as a map from {@link Game}s to {@link Move}. 
     */
    HashMap<Game, Move> curPolicy = new HashMap<>();

    /**
     * The MDP model used, see {@link TTTMDP}.
     */
    TTTMDP mdp;

    /**
     * Discount factor used in value calculations.
     */
    double discount = 0.9;

    /**
     * The (convergence) delta.
     */
    double delta = 0.1;

    /**
     * Constructor: Initializes the agent and trains it using Policy Iteration.
     */
    public PolicyIterationAgent() {
        super();
        this.mdp = new TTTMDP();
        initValues();
        initRandomPolicy();
        train();
    }

    /**
     * Use this constructor to initialize a learning agent with default MDP parameters (rewards, transitions, etc).
     * @param discountFactor The discount factor to use for value calculations.
     */
    public PolicyIterationAgent(double discountFactor) {
        this.discount = discountFactor;
        this.mdp = new TTTMDP();
        initValues();
        initRandomPolicy();
        train();
    }

    /**
     * Use this constructor to set the various parameters of the Tic-Tac-Toe MDP.
     * @param discountFactor Discount factor for value calculations.
     * @param winningReward Reward for winning the game.
     * @param losingReward Penalty for losing the game.
     * @param livingReward Cost for each move (living penalty).
     * @param drawReward Reward for ending the game in a draw.
     */
    public PolicyIterationAgent(double discountFactor, double winningReward, double losingReward, double livingReward, double drawReward) {
        this.discount = discountFactor;
        this.mdp = new TTTMDP(winningReward, losingReward, livingReward, drawReward);
        initValues();
        initRandomPolicy();
        train();
    }

    /**
     * Initializes the {@link #policyValues} map, and sets the initial value of all states to 0.
     */
    public void initValues() {
        List<Game> allGames = Game.generateAllValidGames('X'); // All valid games where it is X's turn, or terminal
        for (Game g : allGames) {
            this.policyValues.put(g, 0.0); // Default value of 0 for all states
        }
    }

    /**
     * Generates a random initial policy by assigning a random valid move to each state.
     * This fills the {@link #curPolicy} map for every state.
     */
    public void initRandomPolicy() {
        // Create a Random object to select moves randomly
        Random random = new Random();
        
        // Iterate through all game states in the policy values map
        for (Game g : this.policyValues.keySet()) {
            // Skip terminal states since they don't require moves
            if (!g.isTerminal()) {
                // Get the list of possible moves for the current state
                List<Move> moves = g.getPossibleMoves();
                
                // Select a random move from the list and assign it to the current policy
                this.curPolicy.put(g, moves.get(random.nextInt(moves.size())));
            }
        }
    }


    /**
     * Performs policy evaluation steps until the maximum change in values is less than {@code delta}.
     * This updates the {@link PolicyIterationAgent#policyValues} map for each state.
     *
     * @param delta The threshold for checking convergence of values.
     */
    protected void evaluatePolicy(double delta) {
        boolean isConverged; // Flag to check if the policy evaluation has converged
        
        do {
            isConverged = true; // Assume convergence unless proven otherwise
            
            // Iterate through all game states in the policy values map
            for (Game g : this.policyValues.keySet()) {
                // Assign a value of 0 to terminal states
                if (g.isTerminal()) {
                    this.policyValues.put(g, 0.0);
                    continue;
                }

                double newValue = 0.0; // Initialize the new value for the current state
                Move action = this.curPolicy.get(g); // Get the current policy's action for the state
                
                // Calculate the new value using the Bellman equation
                for (TransitionProb transition : this.mdp.generateTransitions(g, action)) {
                    newValue += transition.prob * (transition.outcome.localReward + 
                               (discount * this.policyValues.get(transition.outcome.sPrime)));
                }

                // Check if the change in value exceeds the convergence threshold
                if (Math.abs(this.policyValues.get(g) - newValue) > delta) {
                    isConverged = false; // Set to false if values have not stabilized
                }

                // Update the state value in the policy values map
                this.policyValues.put(g, newValue);
            }
        } while (!isConverged); // Continue until values converge
    }

    /**
     * Improves the current policy by finding better moves for each state.
     * This updates the {@link #curPolicy} map if better moves are found.
     *
     * @return true if the policy was improved, false otherwise.
     */
    protected boolean improvePolicy() {
        boolean policyChanged = false; // Flag to indicate if the policy has been updated

        // Iterate through all states in the current policy
        for (Game g : this.curPolicy.keySet()) {
            // Skip terminal states as no actions are needed
            if (g.isTerminal()) {
                continue;
            }

            Move bestMove = null; // Variable to store the best move for the current state
            double bestValue = -Double.MAX_VALUE; // Initialize the best value with a very low value

            // Iterate through all possible moves for the current state
            for (Move m : g.getPossibleMoves()) {
                double moveValue = 0.0; // Initialize the value for the current move

                // Calculate the expected value of taking this move using the Bellman equation
                for (TransitionProb transition : this.mdp.generateTransitions(g, m)) {
                    moveValue += transition.prob * (transition.outcome.localReward + 
                                (discount * this.policyValues.get(transition.outcome.sPrime)));
                }

                // Update the best move if the current move has a higher value
                if (moveValue > bestValue) {
                    bestValue = moveValue;
                    bestMove = m;
                }
            }

            // Check if the best move is different from the current policy's move
            if (!this.curPolicy.get(g).equals(bestMove)) {
                this.curPolicy.put(g, bestMove); // Update the policy with the best move
                policyChanged = true; // Indicate that the policy has been improved
            }
        }

        return policyChanged; // Return true if the policy was updated, false otherwise
    }

    /**
     * Runs policy evaluation and improvement steps until the policy no longer changes.
     * This method alternates between {@link PolicyIterationAgent#evaluatePolicy} and {@link PolicyIterationAgent#improvePolicy}.
     */
    public void train() {
        // Continuously evaluate and improve the policy until no further improvements are possible
        do {
            this.evaluatePolicy(delta); // Evaluate the current policy based on state values
        } while (this.improvePolicy()); // Continue improving the policy until it stops changing

        // Assign the final learned policy to the agent
        super.policy = new Policy(curPolicy); 
    }


    public static void main(String[] args) throws IllegalMoveException {
        /**
         * Test code to run the Policy Iteration Agent against a Human Agent.
         */
        PolicyIterationAgent pi = new PolicyIterationAgent();
        HumanAgent h = new HumanAgent();
        Game g = new Game(pi, h, h);
        g.playOut();
    }
}
