import trace_malloc as trace


'''trace 10 files with maximum memory allocated'''

trace.start()

# ... run your code ...

snapshot = trace.take_snapshot()
top_stats = snapshot.statistics('lineno')

print("[ Top 10 ]")
for stat in top_stats[:10]:
    print(stat)



'''Backtrack the largest memory block'''

# Store 25 frames
trace.start(25)

# ... run your code ...

snapshot = trace.take_snapshot()
top_stats = snapshot.statistics('traceback')

# pick the biggest memory block
stat = top_stats[0]
print("%s memory blocks: %.1f KiB" % (stat.count, stat.size / 1024))
for line in stat.traceback.format():
    print(line)

''' '''