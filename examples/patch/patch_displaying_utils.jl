normpatch(patch) = scaleminmax.(minimum(patch),maximum(patch)).(patch)
viewpatch(patch) =
   [fill(RGB(1,0,0),(1,size(patch,2)+2));
    fill(RGB(1,0,0),(size(patch,1),1)) Gray.(patch) fill(RGB(1,0,0),(size(patch,1),1));
    fill(RGB(1,0,0),(1,size(patch,2)+2))]
viewpatches(patches,zoom=5) = hvcat(size(patches,2),
    permutedims(viewpatch.(repeat.(patches,inner=(zoom,zoom))),[2,1])...)
#;