================================================================
         FACIAL LANDMARKS DATASET â€“ CSV FILE STRUCTURE
================================================================

File name:      FaceGest_landmarks.csv
File name: 		extract_keypoint_sample.py (shows how key points were extrcated)
Header:         NO HEADER ROW
Delimiter:      Comma (,)
Total columns:  1405


Column description:
- Column 1  (index 0) â†’ Integer class label (0â€“12)
- Columns 2â€“1405 (index 1â€“1404) â†’ 1404 flattened facial landmark features
  (MediaPipe Face Mesh 468 landmarks Ã— 3 coordinates = 1404 values:
   x, y, z flattened in MediaPipe order)

================================================================
                  CLASS LABEL MAPPING
================================================================

Label   | Action / Expression                          | Description
--------|----------------------------------------------|------------------------------------
0       | Single blink right                           | Right eye blink only
1       | Single blink left                            | Left eye blink only
2       | Double blink                                 | Both eyes blink twice quickly
3       | Raise eyebrows                               | Both eyebrows raised
4       | Mouth open                                   | Mouth opened (e.g., surprise/"O" shape)
5       | Smile                                        | Genuine/ Duchenne smile
6       | Frown                                        | Sad or displeased frown
7       | Lips pursed                                  | Lips pressed together or puckered
8       | Nodding (up and down)                        | Head nod (yes motion)
9       | Shake head (horizontally)                   | Head shake (no motion)
10      | Blink + Smile                                | Simultaneous eye blink and smile
11      | Raise eyebrows + Mouth open                  | Surprise-like expression
12      | Wink + Head tilt (either direction)          | Wink with any lateral head tilt

================================================================
EXAMPLE OF FIRST FEW ROWS (plain CSV â€“ no header)
================================================================

0,0.512,0.398,0.021,0.489,0.412,0.019,...,0.678,0.745,0.045
1,0.508,0.401,0.022,0.492,0.409,0.020,...,0.681,0.741,0.044
2,0.515,0.395,0.023,0.487,0.415,0.018,...,0.675,0.748,0.046
3,0.510,0.388,0.025,0.491,0.420,0.017,...,0.680,0.752,0.043
...
12,0.520,0.402,0.020,0.485,0.410,0.021,...,0.673,0.739,0.047

================================================================