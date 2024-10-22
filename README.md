Crowd counting in single-view images has achieved outstanding performance on existing counting datasets. However, single-view counting is not applicable to large and wide scenes (eg, public parks, long subway platforms, or event spaces) because a single camera cannot capture the whole scene in adequate detail for counting, eg, when the scene is too large to fit into the field-of-view of the camera, too long so that the resolution is too low on faraway crowds, or when there are too many large objects that occlude large portions of the crowd. 
Therefore, to solve the wide-area counting task requires multiple cameras with overlapping fields-of-view. Here, based on existing research papers on the topic, we have remiplemented a deep neural network framework for multi-view crowd counting, which fuses information from multiple camera views to predict a scene-level density map on the ground-plane of the 3D world. 
We consider 3 versions of the fusion framework: the late fusion model fuses camera-view density map; the naive early fusion model fuses camera-view feature maps; and the multi-view multi-scale early fusion model favors that features aligned to the same ground-plane point have consistent scales. 
Finally, we use the final trained model after model selection on the PETS2009, and check how accurate the results are!

Any edits or suggestions from you are welcome.
Open a pull request in case you want to suggest some modifications of the codebase.

Re-Implemented by:
Ritesh Verma
Fourth-Year Undergraduate
Department of Electrical Engineering
IIT Kanpur