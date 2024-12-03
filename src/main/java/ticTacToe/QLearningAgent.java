package ticTacToe;

import java.util.List;
import java.util.Random;

/**
 * A Q-Learning agent with a Q-Table, i.e. a table of Q-Values. This table is implemented in the {@link QTable} class.
 * 
 * The methods to implement are: 
 * (1) {@link QLearningAgent#train}
 * (2) {@link QLearningAgent#extractPolicy}
 * 
 * Your agent acts in a {@link TTTEnvironment} which provides the method {@link TTTEnvironment#executeMove} which returns an {@link Outcome} object, in other words
 * an [s,a,r,s']: source state, action taken, reward received, and the target state after the opponent has played their move. You may want/need to edit
 * {@link TTTEnvironment} - but you probably won't need to. 
 * @author ae187
 */
public class QLearningAgent extends Agent {
	
	/**
	 * The learning rate, between 0 and 1.
	 */
	double alpha = 0.1;
	
	/**
	 * The number of episodes to train for
	 */
	int numEpisodes = 10000;
	
	/**
	 * The discount factor (gamma)
	 */
	double discount = 0.9;
	
	
	/**
	 * The epsilon in the epsilon greedy policy used during training.
	 */
	double epsilon = 0.1;
	
	/**
	 * This is the Q-Table. To get a value for an (s,a) pair, i.e., a (game, move) pair.
	 */
	QTable qTable = new QTable();
	
	/**
	 * This is the environment the agent interacts with during training.
	 * By default, the opponent is random.
	 */
	TTTEnvironment env = new TTTEnvironment();
	
	/**
	 * Constructs a Q-Learning agent with specific settings.
	 * @param opponent The opponent the agent will play against.
	 * @param learningRate The rate at which the agent learns.
	 * @param numEpisodes The number of training episodes.
	 * @param discount The discount factor for future rewards.
	 */
	public QLearningAgent(Agent opponent, double learningRate, int numEpisodes, double discount) {
		env = new TTTEnvironment(opponent);
		this.alpha = learningRate;
		this.numEpisodes = numEpisodes;
		this.discount = discount;
		initQTable();
		train();
	}
	
	/**
	 * Initializes Q-values for all valid game and move pairs to 0.
	 */
	protected void initQTable() {
		List<Game> allGames = Game.generateAllValidGames('X'); // all valid games where it's X's turn or terminal
		for (Game g : allGames) {
			List<Move> moves = g.getPossibleMoves();
			for (Move m : moves) {
				this.qTable.addQValue(g, m, 0.0); // set initial Q-values to 0
			}
		}
	}
	
	/**
	 * Uses default settings: Random opponent, 0.1 learning rate, and 50000 episodes.
	 */
	public QLearningAgent() {
		this(new RandomAgent(), 0.1, 50000, 0.9);
	}
	
	/**
	 * Trains the agent by playing multiple episodes and updating Q-values using Q-Learning.
	 * Uses epsilon-greedy for move selection: exploration with epsilon probability, exploitation otherwise.
	 * Updates Q-values based on the formula.
	 */
	public void train() {
	    // Run through the specified number of training episodes
	    for (int i = 0; i < numEpisodes; i++) {
	        // Continue playing until a terminal state is reached
	        while (!env.isTerminal()) {
	            Game g = env.getCurrentGameState(); // Get the current game state
	            if (g.isTerminal()) continue; // Skip if the game is already over
	            
	            Move m = pickEpsilonGreedyMove(g); // Choose a move using the epsilon-greedy strategy
	            Outcome outcome = null;

	            try {
	                outcome = env.executeMove(m); // Apply the chosen move in the environment
	            } catch (IllegalMoveException e) {
	                e.printStackTrace(); // Handle any invalid moves
	            }

	            // Retrieve the current Q-value for the state-action pair
	            double currentQ = qTable.getQValue(outcome.s, outcome.move);
	            // Calculate the maximum Q-value for the next state
	            double maxFutureQ = maxQValue(outcome.sPrime);
	            // Update the Q-value using the Q-learning update formula
	            double updatedQ = (1 - alpha) * currentQ + alpha * (outcome.localReward + discount * maxFutureQ);

	            // Store the updated Q-value in the Q-table
	            qTable.addQValue(outcome.s, outcome.move, updatedQ);
	        }
	        env.reset(); // Reset the environment for the next training episode
	    }
	    this.policy = extractPolicy(); // Extract the learned policy after training is complete
	}

	/**
	 * Chooses a move using epsilon-greedy: random with epsilon probability, best Q-value otherwise.
	 * @param g Current game state.
	 * @return Chosen move.
	 */
	private Move pickEpsilonGreedyMove(Game g) {
	    List<Move> moves = g.getPossibleMoves(); // Get all possible moves for the current state
	    Random random = new Random();

	    // Exploration: choose a random move with probability epsilon
	    if (random.nextDouble() < epsilon) {
	        return moves.get(random.nextInt(moves.size())); // Return a randomly chosen move
	    }

	    // Exploitation: choose the move with the highest Q-value
	    double maxQ = Double.NEGATIVE_INFINITY;
	    Move bestMove = null;

	    // Iterate through all possible moves to find the one with the maximum Q-value
	    for (Move m : moves) {
	        double qValue = qTable.getQValue(g, m); // Retrieve the Q-value for the current state and move
	        if (qValue > maxQ) {
	            maxQ = qValue; // Update the maximum Q-value
	            bestMove = m; // Update the best move
	        }
	    }
	    return bestMove; // Return the best move found
	}

	/**
	 * Finds the maximum Q-value for the next state.
	 * @param gPrime Next game state.
	 * @return Maximum Q-value.
	 */
	private double maxQValue(Game gPrime) {
	    // If the game state is terminal, the maximum Q-value is 0
	    if (gPrime.isTerminal()) return 0.0;

	    double maxQ = Double.NEGATIVE_INFINITY; // Initialize with the smallest possible value

	    // Iterate through all possible moves for the given state
	    for (Move m : gPrime.getPossibleMoves()) {
	        double qValue = qTable.getQValue(gPrime, m); // Retrieve the Q-value for this move
	        maxQ = Math.max(maxQ, qValue); // Update maxQ if the current Q-value is greater
	    }
	    return maxQ; // Return the maximum Q-value found
	}


	/** 
	 * Extracts the best moves for each game state based on Q-values.
	 * @return The policy mapping states to moves.
	 */
	public Policy extractPolicy() {
	    Policy p = new Policy(); // Initialize a new policy

	    // Iterate through all game states in the Q-table
	    for (Game g : qTable.keySet()) {
	        if (g.isTerminal()) continue; // Skip terminal states, no policy needed

	        double maxQ = Double.NEGATIVE_INFINITY; // Start with the lowest possible Q-value
	        Move bestMove = null; // Placeholder for the best move

	        // Iterate through all possible moves for the current game state
	        for (Move m : g.getPossibleMoves()) {
	            double qValue = qTable.getQValue(g, m); // Retrieve the Q-value for the (state, move) pair
	            if (qValue > maxQ) { // If this move has a higher Q-value, update maxQ and bestMove
	                maxQ = qValue;
	                bestMove = m;
	            }
	        }
	        p.policy.put(g, bestMove); // Save the best move for this state in the policy
	    }
	    return p; // Return the extracted policy
	}

	/**
	 * Test the agent against a human player.
	 */
	public static void main(String[] args) throws IllegalMoveException {
		QLearningAgent agent = new QLearningAgent();
		HumanAgent human = new HumanAgent();

		Game game = new Game(agent, human, human);
		game.playOut(); // play game
	}
}
