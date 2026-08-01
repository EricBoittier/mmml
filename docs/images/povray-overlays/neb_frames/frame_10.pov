#version 3.7;
global_settings { assumed_gamma 1.0 }
background { color rgbt <1,1,1,1> }
camera { orthographic location <5.577521,3.880014,-13.095049> look_at <0,0,0> right x*8.34203096 up y*6.256523220000001 }
light_source { <-4.850018,9.700036,-9.700036> color rgb <1,1,1> area_light <2,0,0>,<0,2,0>,5,5 adaptive 1 jitter }
light_source { <9.700036,-4.850018,-4.850018> color rgb <0.35,0.40,0.52> }
plane { y, -2.15 pigment { color rgbt <0.91,0.92,0.94,0.82> } finish { diffuse 0.75 } }
cylinder { <-1.127328,0.412272,0.243175>, <0.376462,-0.054537,0.247464>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <-1.127328,0.412272,0.243175>, <-1.233887,1.315711,0.714488>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <-1.127328,0.412272,0.243175>, <-1.725224,-0.265341,0.725959>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <-1.127328,0.412272,0.243175>, <-1.479142,0.514581,-0.713643>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <0.376462,-0.054537,0.247464>, <0.687335,-0.143962,1.288872>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <0.376462,-0.054537,0.247464>, <0.421317,-1.015432,-0.266183>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <0.376462,-0.054537,0.247464>, <0.955677,0.705695,-0.277603>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
sphere { <3.124794,-1.468990,-1.962529>, 0.5304 texture { pigment { color rgb <0.200000,0.720000,0.280000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <-1.127328,0.412272,0.243175>, 0.3692 texture { pigment { color rgb <0.180000,0.320000,0.880000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <0.376462,-0.054537,0.247464>, 0.3952 texture { pigment { color rgb <0.220000,0.240000,0.280000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <-1.233887,1.315711,0.714488>, 0.2400 texture { pigment { color rgb <0.860000,0.870000,0.900000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <-1.725224,-0.265341,0.725959>, 0.2400 texture { pigment { color rgb <0.860000,0.870000,0.900000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <-1.479142,0.514581,-0.713643>, 0.2400 texture { pigment { color rgb <0.860000,0.870000,0.900000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <0.687335,-0.143962,1.288872>, 0.2400 texture { pigment { color rgb <0.860000,0.870000,0.900000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <0.421317,-1.015432,-0.266183>, 0.2400 texture { pigment { color rgb <0.860000,0.870000,0.900000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <0.955677,0.705695,-0.277603>, 0.2400 texture { pigment { color rgb <0.860000,0.870000,0.900000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
cylinder { <3.398916,-1.618369,-1.681562>, <3.776252,-1.823994,-1.294804>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <3.776252,-1.823994,-1.294804>, 0.12, <3.919840,-1.902240,-1.147631>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <-0.782050,0.292490,0.036209>, <-0.543358,0.209685,-0.106868>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <-0.543358,0.209685,-0.106868>, 0.12, <-0.362498,0.146942,-0.215279>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <-0.013611,0.096700,0.210440>, <-0.594787,0.322029,0.155277>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <-0.594787,0.322029,0.155277>, 0.12, <-0.799111,0.401248,0.135883>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <-1.115304,0.921958,0.629069>, <-1.014411,0.586945,0.556393>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <-1.014411,0.586945,0.556393>, 0.12, <-0.952295,0.380693,0.511649>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <-1.316812,-0.199856,0.653083>, <-0.860117,-0.126629,0.571591>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <-0.860117,-0.126629,0.571591>, 0.12, <-0.646186,-0.092327,0.533417>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <-1.621814,0.572731,-0.322921>, <-1.916659,0.692903,0.484548>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <-1.916659,0.692903,0.484548>, 0.12, <-1.991392,0.723362,0.689212>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <0.409918,-0.047033,0.988797>, <-0.336465,0.213753,0.181454>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <-0.336465,0.213753,0.181454>, 0.12, <-0.481779,0.264525,0.024272>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <0.591752,-0.637814,-0.197213>, <0.769258,-0.244531,-0.125381>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <0.769258,-0.244531,-0.125381>, 0.12, <0.858534,-0.046732,-0.089254>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <0.960515,0.298791,-0.381552>, <0.963862,0.017267,-0.453470>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <0.963862,0.017267,-0.453470>, 0.12, <0.966397,-0.195873,-0.507919>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <-1.127328,0.412272,-0.106825>, <0.376462,-0.054537,-0.102536>, 0.04 pigment { color rgb <0.48,0.08,0.72> } finish { emission 0.25 } }
sphere { <-0.375433,0.178868,-0.104680>, 0.075 pigment { color rgb <0.48,0.08,0.72> } }
