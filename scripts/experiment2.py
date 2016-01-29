import amt
t = amt.TurnTableControl()
izzy = amt.PyControl(115200, .04, [0,0,0,0,0],[0,0,0]); # same with this

print izzy.getState()
print t.getState()
print "processing..."
turker = amt.AMT(None, izzy, t, None)
print turker.delta2state([10, 10, 10, 10])

