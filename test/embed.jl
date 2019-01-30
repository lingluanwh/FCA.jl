X = [1 2; 3 4]
@test mat_embed(X, (2,3), zero_pos=[2, 3], tpin="rec", tpout="rec") == [1 0 2; 0 3 4]
@test mat_embed(X, (3,2), zero_pos=[3, 4], tpin="rec", tpout="rec") == [1 0; 3 2; 0 4]
@test mat_embed(X, (4,4), zero_pos=[3, 4], tpin="rec", tpout="her") == [0 1.0 3.0 0.0; 1.0 0 0.0 2.0; 3.0 0.0 0 4.0; 0.0 2.0 4.0 0]

X = [2 3; 3 2]
@test mat_embed(X, (2,3), zero_pos=[1,2,3], tpin="her", tpout="rec") == [0 0 3; 0 2 2]
@test mat_embed(X, (2,2), zero_pos=[4], tpin="her", tpout="rec") == [2 2; 3 0]
@test mat_embed(X, (3,3), zero_pos=[], tpin="her", tpout="her") == [0 2 3; 2 0 2; 3 2 0]