#version 3.7;
global_settings { assumed_gamma 1.0 }
background { color rgbt <1,1,1,1> }
camera { orthographic location <10.215410,7.106372,-23.984005> look_at <0,0,0> right x*15.27869974846231 up y*11.459024811346731 }
light_source { <-8.882965,17.765930,-17.765930> color rgb <1,1,1> area_light <2,0,0>,<0,2,0>,5,5 adaptive 1 jitter }
light_source { <17.765930,-8.882965,-8.882965> color rgb <0.35,0.40,0.52> }
plane { y, -2.15 pigment { color rgbt <0.91,0.92,0.94,0.82> } finish { diffuse 0.75 } }
cylinder { <-3.515518,-1.198078,-2.544434>, <-4.411354,-1.544094,-2.073733>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <-3.515518,-1.198078,-2.544434>, <-3.289571,-0.208392,-2.211854>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <-3.515518,-1.198078,-2.544434>, <-3.727040,-1.150684,-3.592196>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <-3.515518,-1.198078,-2.544434>, <-2.460344,-2.135753,-2.218087>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <-2.460344,-2.135753,-2.218087>, <-2.752080,-3.246121,-1.771247>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <-2.460344,-2.135753,-2.218087>, <-1.152484,-1.766308,-2.387863>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <-1.152484,-1.766308,-2.387863>, <-0.526149,-2.443794,-1.984745>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <-1.152484,-1.766308,-2.387863>, <-0.641824,-0.645382,-3.159673>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <-0.641824,-0.645382,-3.159673>, <-1.397680,-0.355364,-3.888258>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <-0.641824,-0.645382,-3.159673>, <0.618150,-1.036406,-3.918248>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <-0.641824,-0.645382,-3.159673>, <-0.452491,0.645631,-2.336589>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <0.618150,-1.036406,-3.918248>, <0.454600,-1.939316,-4.490630>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <0.618150,-1.036406,-3.918248>, <1.460446,-1.192020,-3.262037>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <0.618150,-1.036406,-3.918248>, <0.879536,-0.242543,-4.597632>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <-0.452491,0.645631,-2.336589>, <-0.715214,1.707474,-2.906799>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <-0.452491,0.645631,-2.336589>, <0.039085,0.717121,-1.036188>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <0.039085,0.717121,-1.036188>, <0.078088,1.697782,-0.781646>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <0.039085,0.717121,-1.036188>, <0.607117,-0.176684,0.025674>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <0.607117,-0.176684,0.025674>, <1.179861,0.529628,0.634480>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <0.607117,-0.176684,0.025674>, <1.564700,-1.263466,-0.398361>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <0.607117,-0.176684,0.025674>, <-0.552158,-0.712654,0.905029>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <1.564700,-1.263466,-0.398361>, <1.900630,-1.753491,0.498678>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <1.564700,-1.263466,-0.398361>, <1.119192,-2.014060,-1.023105>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <1.564700,-1.263466,-0.398361>, <2.422759,-0.849470,-0.896210>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <-0.552158,-0.712654,0.905029>, <-1.238993,-1.661317,0.506732>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <-0.552158,-0.712654,0.905029>, <-0.954183,-0.014899,2.025779>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <-0.954183,-0.014899,2.025779>, <-1.775331,-0.461779,2.394429>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <-0.954183,-0.014899,2.025779>, <-0.273671,0.815039,3.051965>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <-0.273671,0.815039,3.051965>, <-0.605222,0.386029,3.999698>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <-0.273671,0.815039,3.051965>, <-0.749996,2.249270,2.965053>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <-0.273671,0.815039,3.051965>, <1.259178,0.605730,3.123910>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <-0.749996,2.249270,2.965053>, <-0.438438,2.724097,2.044282>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <-0.749996,2.249270,2.965053>, <-0.385445,2.814327,3.806565>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <-0.749996,2.249270,2.965053>, <-1.830885,2.269744,3.006626>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <1.259178,0.605730,3.123910>, <1.784361,-0.495613,3.274230>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <1.259178,0.605730,3.123910>, <2.001584,1.773361,3.049752>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <2.001584,1.773361,3.049752>, <1.576526,2.686963,2.942771>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <2.001584,1.773361,3.049752>, <3.416463,1.736444,3.261259>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <3.416463,1.736444,3.261259>, <3.692718,1.464063,4.285333>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <3.416463,1.736444,3.261259>, <3.893987,0.987476,2.624982>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <3.416463,1.736444,3.261259>, <3.897091,2.697508,3.052309>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
sphere { <-3.515518,-1.198078,-2.544434>, 0.3952 texture { pigment { color rgb <0.220000,0.240000,0.280000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <-4.411354,-1.544094,-2.073733>, 0.2400 texture { pigment { color rgb <0.860000,0.870000,0.900000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <-3.289571,-0.208392,-2.211854>, 0.2400 texture { pigment { color rgb <0.860000,0.870000,0.900000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <-3.727040,-1.150684,-3.592196>, 0.2400 texture { pigment { color rgb <0.860000,0.870000,0.900000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <-2.460344,-2.135753,-2.218087>, 0.3952 texture { pigment { color rgb <0.220000,0.240000,0.280000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <-2.752080,-3.246121,-1.771247>, 0.3432 texture { pigment { color rgb <0.860000,0.160000,0.130000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <-1.152484,-1.766308,-2.387863>, 0.3692 texture { pigment { color rgb <0.180000,0.320000,0.880000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <-0.526149,-2.443794,-1.984745>, 0.2400 texture { pigment { color rgb <0.860000,0.870000,0.900000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <-0.641824,-0.645382,-3.159673>, 0.3952 texture { pigment { color rgb <0.220000,0.240000,0.280000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <-1.397680,-0.355364,-3.888258>, 0.2400 texture { pigment { color rgb <0.860000,0.870000,0.900000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <0.618150,-1.036406,-3.918248>, 0.3952 texture { pigment { color rgb <0.220000,0.240000,0.280000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <0.454600,-1.939316,-4.490630>, 0.2400 texture { pigment { color rgb <0.860000,0.870000,0.900000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <1.460446,-1.192020,-3.262037>, 0.2400 texture { pigment { color rgb <0.860000,0.870000,0.900000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <0.879536,-0.242543,-4.597632>, 0.2400 texture { pigment { color rgb <0.860000,0.870000,0.900000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <-0.452491,0.645631,-2.336589>, 0.3952 texture { pigment { color rgb <0.220000,0.240000,0.280000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <-0.715214,1.707474,-2.906799>, 0.3432 texture { pigment { color rgb <0.860000,0.160000,0.130000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <0.039085,0.717121,-1.036188>, 0.3692 texture { pigment { color rgb <0.180000,0.320000,0.880000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <0.078088,1.697782,-0.781646>, 0.2400 texture { pigment { color rgb <0.860000,0.870000,0.900000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <0.607117,-0.176684,0.025674>, 0.3952 texture { pigment { color rgb <0.220000,0.240000,0.280000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <1.179861,0.529628,0.634480>, 0.2400 texture { pigment { color rgb <0.860000,0.870000,0.900000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <1.564700,-1.263466,-0.398361>, 0.3952 texture { pigment { color rgb <0.220000,0.240000,0.280000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <1.900630,-1.753491,0.498678>, 0.2400 texture { pigment { color rgb <0.860000,0.870000,0.900000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <1.119192,-2.014060,-1.023105>, 0.2400 texture { pigment { color rgb <0.860000,0.870000,0.900000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <2.422759,-0.849470,-0.896210>, 0.2400 texture { pigment { color rgb <0.860000,0.870000,0.900000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <-0.552158,-0.712654,0.905029>, 0.3952 texture { pigment { color rgb <0.220000,0.240000,0.280000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <-1.238993,-1.661317,0.506732>, 0.3432 texture { pigment { color rgb <0.860000,0.160000,0.130000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <-0.954183,-0.014899,2.025779>, 0.3692 texture { pigment { color rgb <0.180000,0.320000,0.880000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <-1.775331,-0.461779,2.394429>, 0.2400 texture { pigment { color rgb <0.860000,0.870000,0.900000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <-0.273671,0.815039,3.051965>, 0.3952 texture { pigment { color rgb <0.220000,0.240000,0.280000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <-0.605222,0.386029,3.999698>, 0.2400 texture { pigment { color rgb <0.860000,0.870000,0.900000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <-0.749996,2.249270,2.965053>, 0.3952 texture { pigment { color rgb <0.220000,0.240000,0.280000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <-0.438438,2.724097,2.044282>, 0.2400 texture { pigment { color rgb <0.860000,0.870000,0.900000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <-0.385445,2.814327,3.806565>, 0.2400 texture { pigment { color rgb <0.860000,0.870000,0.900000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <-1.830885,2.269744,3.006626>, 0.2400 texture { pigment { color rgb <0.860000,0.870000,0.900000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <1.259178,0.605730,3.123910>, 0.3952 texture { pigment { color rgb <0.220000,0.240000,0.280000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <1.784361,-0.495613,3.274230>, 0.3432 texture { pigment { color rgb <0.860000,0.160000,0.130000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <2.001584,1.773361,3.049752>, 0.3692 texture { pigment { color rgb <0.180000,0.320000,0.880000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <1.576526,2.686963,2.942771>, 0.2400 texture { pigment { color rgb <0.860000,0.870000,0.900000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <3.416463,1.736444,3.261259>, 0.3952 texture { pigment { color rgb <0.220000,0.240000,0.280000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <3.692718,1.464063,4.285333>, 0.2400 texture { pigment { color rgb <0.860000,0.870000,0.900000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <3.893987,0.987476,2.624982>, 0.2400 texture { pigment { color rgb <0.860000,0.870000,0.900000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <3.897091,2.697508,3.052309>, 0.2400 texture { pigment { color rgb <0.860000,0.870000,0.900000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
cylinder { <-3.599792,-1.338613,-2.157720>, <-3.567290,-1.284412,-2.306866>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <-3.567290,-1.284412,-2.306866>, 0.12, <-3.611433,-1.358026,-2.104302>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <-4.352539,-1.865617,-1.809984>, <-4.355212,-1.851005,-1.821970>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <-4.355212,-1.851005,-1.821970>, 0.12, <-4.324404,-2.019422,-1.683816>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <-3.144835,0.181839,-2.155540>, <-3.045500,0.449659,-2.116891>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <-3.045500,0.449659,-2.116891>, 0.12, <-2.969686,0.654065,-2.087394>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <-3.454185,-1.153703,-3.911478>, <-3.418693,-1.154095,-3.953009>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <-3.418693,-1.154095,-3.953009>, 0.12, <-3.275769,-1.155676,-4.120251>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <-2.070385,-2.181774,-2.367131>, <-2.053079,-2.183817,-2.373746>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <-2.053079,-2.183817,-2.373746>, 0.12, <-1.848815,-2.207923,-2.451816>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <-2.734685,-2.869313,-1.586551>, <-2.736815,-2.915447,-1.609165>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <-2.736815,-2.915447,-1.609165>, 0.12, <-2.727703,-2.718071,-1.512419>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <-1.357581,-1.494468,-2.633705>, <-1.323615,-1.539487,-2.592992>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <-1.323615,-1.539487,-2.592992>, 0.12, <-1.431047,-1.397094,-2.721767>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <-0.356379,-2.722857,-1.720734>, <-0.323045,-2.777651,-1.668895>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <-0.323045,-2.777651,-1.668895>, 0.12, <-0.234117,-2.923827,-1.530603>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <-0.986221,-0.640746,-2.919321>, <-1.616113,-0.632267,-2.479724>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <-1.616113,-0.632267,-2.479724>, 0.12, <-1.796512,-0.629838,-2.353825>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <-1.000760,-0.491452,-3.869970>, <-1.077097,-0.465279,-3.873487>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <-1.077097,-0.465279,-3.873487>, 0.12, <-0.869186,-0.536563,-3.863908>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <0.352158,-1.019557,-3.593649>, <0.417273,-1.023682,-3.673111>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <0.417273,-1.023682,-3.673111>, 0.12, <0.277944,-1.014856,-3.503083>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <0.297944,-2.263507,-4.706869>, <0.284775,-2.290759,-4.725046>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <0.284775,-2.290759,-4.725046>, 0.12, <0.202718,-2.460573,-4.838314>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <1.799650,-1.253767,-3.022186>, <1.888990,-1.270030,-2.959014>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <1.888990,-1.270030,-2.959014>, 0.12, <2.066668,-1.302374,-2.833378>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <0.980172,0.045757,-4.885999>, <1.038124,0.211777,-5.052056>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <1.038124,0.211777,-5.052056>, 0.12, <1.090838,0.362791,-5.203105>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <-0.117907,0.425559,-2.463168>, <-0.042405,0.375897,-2.491731>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <-0.042405,0.375897,-2.491731>, 0.12, <0.132854,0.260621,-2.558034>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <-1.058445,1.538600,-2.733380>, <-0.990090,1.572232,-2.767917>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <-0.990090,1.572232,-2.767917>, 0.12, <-1.169878,1.483774,-2.677079>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <0.124099,0.307237,-1.070360>, <0.169798,0.086906,-1.088728>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <0.169798,0.086906,-1.088728>, 0.12, <0.214330,-0.127796,-1.106628>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <0.086998,1.365067,-0.525480>, <0.087543,1.344720,-0.509814>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <0.087543,1.344720,-0.509814>, 0.12, <0.092210,1.170440,-0.375632>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <0.393622,0.171529,0.123487>, <-0.180779,1.108390,0.386650>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <-0.180779,1.108390,0.386650>, 0.12, <-0.292609,1.290788,0.437885>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <1.467420,0.472178,0.935162>, <1.552454,0.455189,1.024076>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <1.552454,0.455189,1.024076>, 0.12, <1.703080,0.425096,1.181576>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <1.211300,-1.177829,-0.188183>, <1.180149,-1.170280,-0.169656>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <1.180149,-1.170280,-0.169656>, 0.12, <0.995034,-1.125422,-0.059563>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <2.073749,-1.913315,0.846364>, <2.166254,-1.998716,1.032148>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <2.166254,-1.998716,1.032148>, 0.12, <2.256936,-2.082433,1.214270>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <0.830390,-2.247325,-1.219526>, <0.687529,-2.362714,-1.316690>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <0.687529,-2.362714,-1.316690>, 0.12, <0.536251,-2.484900,-1.419577>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <2.737045,-0.693627,-1.127158>, <2.973058,-0.576597,-1.300588>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <2.973058,-0.576597,-1.300588>, 0.12, <3.137685,-0.494965,-1.421561>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <-0.672691,-0.438118,0.610917>, <-0.803148,-0.140979,0.292592>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <-0.803148,-0.140979,0.292592>, 0.12, <-0.866284,0.002825,0.138534>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <-0.958469,-1.971632,0.469170>, <-0.854328,-2.086832,0.455226>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <-0.854328,-2.086832,0.455226>, 0.12, <-0.707387,-2.249378,0.435551>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <-0.542847,-0.076602,2.084050>, <0.015391,-0.160341,2.163132>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <0.015391,-0.160341,2.163132>, 0.12, <0.230853,-0.192661,2.193656>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <-2.116301,-0.584892,2.606520>, <-2.261725,-0.637400,2.696977>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <-2.261725,-0.637400,2.696977>, 0.12, <-2.440328,-0.701889,2.808072>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <-0.245447,0.429811,2.887032>, <-0.193871,-0.274146,2.585635>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <-0.193871,-0.274146,2.585635>, 0.12, <-0.179087,-0.475932,2.499242>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <-0.784965,0.735749,4.147307>, <-0.756016,0.679423,4.123534>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <-0.756016,0.679423,4.123534>, 0.12, <-0.850167,0.862610,4.200853>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <-0.690748,1.902106,2.736215>, <-0.686368,1.876439,2.719297>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <-0.686368,1.876439,2.719297>, 0.12, <-0.655333,1.694591,2.599430>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <-0.312193,2.857263,1.666487>, <-0.274045,2.897502,1.552330>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <-0.274045,2.897502,1.552330>, 0.12, <-0.207917,2.967256,1.354437>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <-0.273093,3.092114,4.100863>, <-0.177394,3.328724,4.351537>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <-0.177394,3.328724,4.351537>, 0.12, <-0.118543,3.474231,4.505693>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <-2.243471,2.348002,3.013624>, <-2.337519,2.365840,3.015219>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <-2.337519,2.365840,3.015219>, 0.12, <-2.553635,2.406832,3.018885>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <0.841896,0.586779,3.080128>, <0.562915,0.574110,3.050857>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <0.562915,0.574110,3.050857>, 0.12, <0.344339,0.564183,3.027924>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <1.601932,-0.233552,3.001386>, <1.510675,-0.102462,2.864902>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <1.510675,-0.102462,2.864902>, 0.12, <1.415117,0.034808,2.721983>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <2.326976,1.952989,2.854166>, <2.500363,2.048705,2.749947>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <2.500363,2.048705,2.749947>, 0.12, <2.670807,2.142795,2.647497>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <1.668629,2.292359,3.053246>, <1.677524,2.254251,3.063914>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <1.677524,2.254251,3.063914>, 0.12, <1.725768,2.047554,3.121782>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <3.013046,1.700206,3.150167>, <2.954938,1.694986,3.134165>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <2.954938,1.694986,3.134165>, 0.12, <2.743624,1.676005,3.075975>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <3.496578,1.236076,4.578507>, <3.557727,1.307154,4.487106>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <3.557727,1.307154,4.487106>, 0.12, <3.454987,1.187733,4.640673>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <3.856779,0.635261,2.399237>, <3.860020,0.665945,2.418904>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <3.860020,0.665945,2.418904>, 0.12, <3.840531,0.481452,2.300656>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <4.143417,3.031144,2.985893>, <4.098104,2.969770,2.998111>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <4.098104,2.969770,2.998111>, 0.12, <4.227132,3.144531,2.963321>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
