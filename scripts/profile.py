import pstats
p = pstats.Stats('output.pstats')
p.sort_stats('cumulative').print_stats(10) # for top 10 cumulative time
