# N-Tuple Network

N-Tuple Network is a set of patterns. Many board games can be splitted into various patterns, for example Othello can be splitted into 49 2x2 squares, each such square may hold one of 81 (3\*\*4) possible states, or some 8x1 rectangles so each rectangle can hold one of 6561 (3\*\*8) states. Each pattern contains some value, and given position, the sum of all values of patterns may become a board evaluation. N-Tuple Network is function approximator, much like neural network, and it can be trained like neural network.

More on N-Tuples (mostly in reinforcement learning) are in:
[`Learning to Play Othello
with N-Tuple Systems`](https://www.semanticscholar.org/paper/Learning-to-Play-Othello-with-N-Tuple-Systems-Lucas/58fc891bd082eafabb78ebca42f35d3e1e494516)
[`Temporal Difference Learning of N-Tuple Networks
for the Game 2048`](https://ieeexplore.ieee.org/document/6932907/)
[`Reinforcement Learning with N-tuples on the Game Connect-4`](https://www.researchgate.net/publication/235219697_Reinforcement_Learning_with_N-tuples_on_the_Game_Connect-4)
[`Apparently checkers folks discovered N-tuples as well`](http://www.fierz.ch/cake186.php)

In this article I will show how this network can be trained for tic-tac-toe, so that 1-ply search bot will become a perfect player, despite only searching 1 ply ahead instead of all possible states until the end of game.

# N-Tuple Network for tic tac toe

The goal of tic tac toe is to get 3 in row of own pieces - vertically, horizontally or diagonally. I think naturally the N-Tuple Network for such game should be the all eight possible rows of 3. 3 horizontal rows, 3 vertical rows, 1 diagonal and 1 anti-diagonal.

(Image/Code)

Each state of row can be encoded in LUT (lookup table) as index. In this case we have base3 system, empty being 0, X being 1 and O being 2.

(Image/Code)

This way each row has 27 (3\*\*3) possible states. But for many games, to know which side is to move now, is very crucial. In tic tac toe consider there is a row XX_. For player X it is quite advantageous. But if we knew the side to move now we could predict more accurately how good it is for X. If the current player is X, this is clearly a win. If the current player is O, O must play in this row, unless O has its row like _OO. So let's add side to move information to the network. Either we can use 2 separate N-Tuple Networks for side to move, or double the possible states for each row, which is effectively the same. I chose the latter. That is, if side to move is player O, I add 27 to the indexes.

(Image/Code)

And that's our N-Tuple Network! Each position has 8 tuples set. Example:

(Example 1)

(Example 2)



Now having that list of tuples, we can put it to the network to predict the value.
```c++
float weights[8][54];
float bias;

float predict(List<int> tuples) {
    float output = 0;
    for (int i=0; i < tuples.size(); i++) {
        output += weights[i][tuples[i]];
    }
    return tanh(output + bias); // why there's bias? ask some smart mathematician
}
```

As activation function (a word from neural network world), I used tanh. It has nice output of range (-1,1), so -1 would mean loss and +1 win. Divide by 2 and add 0.5, you'll get range (0,1) and can intepret it as probability of winning.

**Note:** This network is from persepctive of player X (X wins = 1, O wins = -1). When used in minimax bot, the player O would take the minus of the predicted output.

# Training with data

There are several ways to train this network. This could be one of reinforcement learning like temporal difference learning, or supervised learning based on labeled examples. Tic tac toe is a solved game. Moreover, it is strongly solved game and we know true value (win, loss, draw) for each possible position in game. So the training will be supervised learning based on true values of positions.

For update we're going to use good old (stochastic) gradient descent. There are many resources for gradient descent with fancy math, but who needs math when you have the pseudocode.

```c++
float alpha = 0.01; // learning rate

float tanh_prime(float x) { // x was already tanhed
    return 1 - x*x;
}

float learn(List<int> tuples, int target) {
    float output = 0;
    for (int i=0; i < tuples.size(); i++) {
        output += weights[i][tuples[i]];
    }
    output = tanh(output + bias);
    float error = target - output;
    float delta = error * tanh_prime(output); // why derivative? again, those pesky mathematicians!
    for (int i=0; i < tuples.size(); i++) {
        weights[i][tuples[i]] += alpha * delta;
    }
    bias += alpha * delta;
}
```

Generally speaking, we feed training data, the positions and their values, into the network so it can adjust its weights to minimize the error in the prediction. In training data, the positions with XXX rows will have value of 1, so likely the network will adjust weights for XXX to be relatively big. 

```c++
void training() {
    List<Pair<List<int>,float>> trainingData = solveAllPositions(); // there are 5478 reachable positions in tic tac toe
    
    for (int epoch = 0; epochs < 100; epochs++) {
        for (Pair<List<int>,float> pair : trainingData) {
            List<int> tuples = pair.first;
            float target = pair.second;
            learn(tuples, target);
        }
        float error = 0;
        for (Pair<List<int>,float> pair : trainingData) {
            List<int> tuples = pair.first;
            float target = pair.second;
            float value = predict(tuples);
            error += (target-value) * (target-value);
        }
        print("Error: " + (error / trainingData.size()));
    }
}
```

This pseudocode will feed the entire data into network for 100 epochs and it will print the mean squared error after each learning pass to see how much the error is decreasing. Error (or cost or loss), epochs, learning rate are some words used in the neural network community and they have the same meaning here. Why learning rate is 0.01 or why epochs is 100 is out of scope this article, but I chose those numbers because they work here.

# Using trained network

The trained N-Tuple Network, which we believe is pretty good at evaluating game positions, can be used as evaluation function for the minimax bot. Normally perfect tic tac toe minimax bots use brute force search until the end of game without evaluation except for win/lose/draw situations. Tic tac toe isn't big game so it's possible. But my experiments show that with this network, trained only for few seconds, and search with only 1 ply, it is sufficient act like perfect bot. 

This is output of [`my example program`](https://github.com/jdermont/tictactoe-ntuple):
```
generating data from games...
there are 5478 positions
training network...
epoch 5. error: 0.123647
epoch 10. error: 0.0952127
epoch 15. error: 0.0842701
epoch 20. error: 0.0790482
epoch 25. error: 0.0744271
epoch 30. error: 0.0719857
epoch 35. error: 0.0703368
epoch 40. error: 0.0690381
epoch 45. error: 0.0678587
epoch 50. error: 0.066961
epoch 55. error: 0.0663684
epoch 60. error: 0.0655014
epoch 65. error: 0.0651461
epoch 70. error: 0.0650244
epoch 75. error: 0.0643357
epoch 80. error: 0.0643997
epoch 85. error: 0.0635387
epoch 90. error: 0.0632871
epoch 95. error: 0.0630048
epoch 100. error: 0.062984
perfect cpu vs random: 
CPU1 CPU2 DRAW: 
86 0 14
177 0 23
268 0 32
351 0 49
440 0 60
532 0 68
618 0 82
707 0 93
793 0 107
878 0 122
network cpu vs random: 
CPU1 CPU2 DRAW: 
95 0 5
188 0 12
281 0 19
371 0 29
468 0 32
563 0 37
654 0 46
744 0 56
841 0 59
936 0 64
network cpu vs cpu ply 3: 
CPU1 CPU2 DRAW: 
29 0 71
49 0 151
85 0 215
113 0 287
142 0 358
177 0 423
205 0 495
234 0 566
260 0 640
281 0 719
network cpu vs perfect cpu: 
CPU1 CPU2 DRAW: 
0 0 100
0 0 200
0 0 300
0 0 400
0 0 500
0 0 600
0 0 700
0 0 800
0 0 900
0 0 1000
```

The network bot never loses against perfect bot, it wins against inferior bots. What's interesting is that against random opponent, it wins more than perfect bot against it! I think this is due to fact that as side effect, network bot makes moves that maximize the set of losing moves for opponent, i.e. it always chooses the center. Perfect bot knows all first moves for X are draws so it chooses one randomly. This may slightly decrease chance of failing for the random bot.



