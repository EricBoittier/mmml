#version 3.7;
global_settings { assumed_gamma 1.0 }
background { color rgbt <1,1,1,1> }
camera { orthographic location <6.514387,4.531748,-15.294648> look_at <0,0,0> right x*9.743257519631257 up y*7.307443139723442 }
light_source { <-5.664685,11.329369,-11.329369> color rgb <1,1,1> area_light <2,0,0>,<0,2,0>,5,5 adaptive 1 jitter }
light_source { <11.329369,-5.664685,-5.664685> color rgb <0.35,0.40,0.52> }
plane { y, -2.15 pigment { color rgbt <0.91,0.92,0.94,0.82> } finish { diffuse 0.75 } }
cylinder { <-1.180170,-2.246530,-1.201688>, <-0.803785,-2.820635,-0.562718>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <-1.180170,-2.246530,-1.201688>, <-1.398409,-3.108164,-1.402627>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <0.969227,1.829620,-1.969818>, <0.453872,1.591560,-2.706969>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <0.969227,1.829620,-1.969818>, <1.309649,2.246113,-2.730827>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <2.457590,-0.974090,1.401209>, <1.882820,-0.568212,1.985379>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <2.457590,-0.974090,1.401209>, <2.890625,-0.237460,1.764792>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <-1.963155,1.003625,1.944878>, <-1.844206,1.799687,1.517308>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <-1.963155,1.003625,1.944878>, <-2.774059,1.484487,1.961079>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
sphere { <-1.180170,-2.246530,-1.201688>, 0.3432 texture { pigment { color rgb <0.860000,0.160000,0.130000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <-0.803785,-2.820635,-0.562718>, 0.2400 texture { pigment { color rgb <0.860000,0.870000,0.900000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <-1.398409,-3.108164,-1.402627>, 0.2400 texture { pigment { color rgb <0.860000,0.870000,0.900000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <0.969227,1.829620,-1.969818>, 0.3432 texture { pigment { color rgb <0.860000,0.160000,0.130000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <0.453872,1.591560,-2.706969>, 0.2400 texture { pigment { color rgb <0.860000,0.870000,0.900000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <1.309649,2.246113,-2.730827>, 0.2400 texture { pigment { color rgb <0.860000,0.870000,0.900000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <2.457590,-0.974090,1.401209>, 0.3432 texture { pigment { color rgb <0.860000,0.160000,0.130000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <1.882820,-0.568212,1.985379>, 0.2400 texture { pigment { color rgb <0.860000,0.870000,0.900000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <2.890625,-0.237460,1.764792>, 0.2400 texture { pigment { color rgb <0.860000,0.870000,0.900000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <-1.963155,1.003625,1.944878>, 0.3432 texture { pigment { color rgb <0.860000,0.160000,0.130000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <-1.844206,1.799687,1.517308>, 0.2400 texture { pigment { color rgb <0.860000,0.870000,0.900000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <-2.774059,1.484487,1.961079>, 0.2400 texture { pigment { color rgb <0.860000,0.870000,0.900000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
cylinder { <-1.137205,-1.828738,-1.203715>, <-1.021608,-0.704679,-1.209170>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <-1.021608,-0.704679,-1.209170>, 0.12, <-0.999103,-0.485836,-1.210232>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <-0.614213,-3.030436,-0.252161>, <-0.526951,-3.127011,-0.109207>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <-0.526951,-3.127011,-0.109207>, 0.12, <-0.427651,-3.236907,0.053466>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <-1.518793,-3.493807,-1.517461>, <-1.824876,-4.474322,-1.809433>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <-1.824876,-4.474322,-1.809433>, 0.12, <-1.887934,-4.676325,-1.869584>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <0.928998,1.879224,-2.384934>, <0.853501,1.972315,-3.163969>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <0.853501,1.972315,-3.163969>, 0.12, <0.832428,1.998298,-3.381410>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <0.718427,1.745837,-2.419550>, <1.028355,1.926574,-2.082835>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <1.028355,1.926574,-2.082835>, 0.12, <1.166931,2.007386,-1.932282>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <1.107517,2.034758,-2.429378>, <0.832847,1.747555,-2.019751>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <0.832847,1.747555,-2.019751>, 0.12, <0.726968,1.636845,-1.861849>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <2.841352,-0.966679,1.230705>, <2.762689,-0.968198,1.265655>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <2.762689,-0.968198,1.265655>, 0.12, <2.963707,-0.964316,1.176343>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <2.115481,-0.275326,2.176393>, <2.083106,-0.316081,2.149813>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <2.083106,-0.316081,2.149813>, 0.12, <2.204977,-0.162664,2.249868>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <2.546012,-0.474760,1.801271>, <2.598368,-0.438708,1.795729>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <2.598368,-0.438708,1.795729>, 0.12, <2.417856,-0.563008,1.814837>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <-2.335963,1.195575,1.921030>, <-2.374163,1.215243,1.918586>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <-2.374163,1.215243,1.918586>, 0.12, <-2.569444,1.315789,1.906094>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <-2.249423,1.854737,1.613064>, <-2.136027,1.839332,1.586268>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <-2.136027,1.839332,1.586268>, 0.12, <-2.348283,1.868168,1.636426>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <-2.375734,1.364341,1.903620>, <-2.262644,1.330229,1.887306>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <-2.262644,1.330229,1.887306>, 0.12, <-2.053997,1.267296,1.857208>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
