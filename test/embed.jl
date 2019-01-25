X = [1 2; 3 4]
@test mat_embed(X, (2,3), zero_pos=[2, 3], in_type="rec", out_type="rec") == [1 0 2; 0 3 4]
@test mat_embed(X, (3,2), zero_pos=[3, 4], in_type="rec", out_type="rec") == [1 0; 3 2; 0 4]
@test mat_embed(X, (4,4), zero_pos=[3, 4], in_type="rec", out_type="her") == [0 1.0 3.0 0.0; 1.0 0 0.0 2.0; 3.0 0.0 0 4.0; 0.0 2.0 4.0 0]

