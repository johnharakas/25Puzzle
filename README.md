# 25Puzzle
25 (5x5) sliding puzzle solver. 

For this variant, even IDA* can take quite a while to find a solution. Among other methods, there are pattern databases which need to be precomputed beforehand. In this case, I'm solving the puzzle in a partial order. Although its relatively straightforward to code the explicit actions needed to move a tile to it's correct slot, it didn't seem too much fun. Instead I've partitioned the search into a search for multiple subgoals (divide and conquer). An upside to this approach is that tiles in their correct states can be omitted from future searches - reducing the search space tremendously. The main drawback is that although the solution is locally optimal, it will not return a globablly optimal solution. This is in part because I use a specific set of "heuristics" to guide the search:

![equation](https://latex.codecogs.com/gif.latex?%5Cinline%20d_1%28t_i%2C%20t_0%29%20&plus;%20d_1%28t_i%2C%20g_i%29) 

which is the manhattan distance from the target tile i to the goal and the manhattan distance between the target tile and the blank tile. Since I am not explicitly guiding the search by tossing out actions that don't involve the target, this ensures that actions performed near or on the target tile are less costly.

Once the first 3 tiles are lined up, the heuristics change a bit. In order for (4,5) to get into place, they need to be adjacent to one-another and in a specific order. The goal state can only be reached by branching from a couple of very specific states so the search is guided to a target state such as the ones below. 

|   |   |   |   |   |
|---|---|---|---|---|
| 1 | 2 | 3 | x | x |
| x | x | x | x | 4 |
| x | x | x | x | 5 |
| x | x | x | x | x |

|   |   |   |   |   |
|---|---|---|---|---|
| 1 | 2 | 3 | x | x |
| x | x | x | 5 | x |
| x | x | x | 4 | x |
| x | x | x | x | x |

To reach a partial goal state where the first row is completed, another set of "heuristics" are used:
![equation](https://latex.codecogs.com/gif.latex?%5Cinline%20d_1%28t_4%2C%20t_5%29%20&plus;%20d_1%28t_4%2C%20g_4%29%20&plus;%20d_1%28t_5%2C%20g_5%29%20&plus;%20d_1%28t_0%2C%20t_4%29%20&plus;%20d_1%28t_0%2C%20t_5%29)

Although I haven't fully determined the admissibility of these heuristics, they do work well. The cost is calculated by the distance between tiles 4 and 5, the distance they are from the goal state, and the distance they are from the blank tile. In other words, it costs more to move 4 away from 5, move away from (4,5),  and move (4,5) away from their goal states. 

After the first row is completed, the process is essentially repeated until the puzzle is completely solved. For nearly all random states, this has been much less than 1s and returns a solution sequence of typically 300 actions to reach a goal state. 
