#version 3.7;
global_settings { assumed_gamma 1.0 }
background { color rgbt <1,1,1,1> }
camera { orthographic location <5.438770,3.783492,-12.769286> look_at <0,0,0> right x*8.1345078 up y*6.10088085 }
light_source { <-4.729365,9.458730,-9.458730> color rgb <1,1,1> area_light <2,0,0>,<0,2,0>,5,5 adaptive 1 jitter }
light_source { <9.458730,-4.729365,-4.729365> color rgb <0.35,0.40,0.52> }
plane { y, -2.15 pigment { color rgbt <0.91,0.92,0.94,0.82> } finish { diffuse 0.75 } }
cylinder { <2.859402,-1.204627,0.218793>, <1.166343,-0.492275,0.086592>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <-1.427521,0.540929,-0.283148>, <-1.857382,-0.111855,0.380590>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <-1.427521,0.540929,-0.283148>, <-1.869963,0.637754,-1.207966>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <-1.427521,0.540929,-0.283148>, <-1.325288,1.414713,0.174111>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <1.166343,-0.492275,0.086592>, <0.751920,-0.301390,-0.836654>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <1.166343,-0.492275,0.086592>, <1.212316,0.581717,0.592733>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <1.166343,-0.492275,0.086592>, <0.490177,-1.064967,0.874945>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
sphere { <2.859402,-1.204627,0.218793>, 0.5304 texture { pigment { color rgb <0.200000,0.720000,0.280000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <-1.427521,0.540929,-0.283148>, 0.3692 texture { pigment { color rgb <0.180000,0.320000,0.880000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <1.166343,-0.492275,0.086592>, 0.3952 texture { pigment { color rgb <0.220000,0.240000,0.280000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <-1.857382,-0.111855,0.380590>, 0.2400 texture { pigment { color rgb <0.860000,0.870000,0.900000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <-1.869963,0.637754,-1.207966>, 0.2400 texture { pigment { color rgb <0.860000,0.870000,0.900000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <-1.325288,1.414713,0.174111>, 0.2400 texture { pigment { color rgb <0.860000,0.870000,0.900000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <0.751920,-0.301390,-0.836654>, 0.2400 texture { pigment { color rgb <0.860000,0.870000,0.900000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <1.212316,0.581717,0.592733>, 0.2400 texture { pigment { color rgb <0.860000,0.870000,0.900000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <0.490177,-1.064967,0.874945>, 0.2400 texture { pigment { color rgb <0.860000,0.870000,0.900000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
cylinder { <3.133524,-1.354007,0.499761>, <3.510860,-1.559631,0.886518>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <3.510860,-1.559631,0.886518>, 0.12, <3.654448,-1.637877,1.033691>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <-1.082243,0.421147,-0.490114>, <-0.843551,0.338341,-0.633190>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <-0.843551,0.338341,-0.633190>, 0.12, <-0.662691,0.275598,-0.741601>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <0.776270,-0.341039,0.049568>, <0.195094,-0.115709,-0.005595>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <0.195094,-0.115709,-0.005595>, 0.12, <-0.009230,-0.036490,-0.024989>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <-1.738799,-0.505608,0.295171>, <-1.637906,-0.840621,0.222495>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <-1.637906,-0.840621,0.222495>, 0.12, <-1.575790,-1.046873,0.177752>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <-1.461551,0.703239,-1.280842>, <-1.004856,0.776466,-1.362334>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <-1.004856,0.776466,-1.362334>, 0.12, <-0.790925,0.810767,-1.400507>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <-1.467960,1.472863,0.564833>, <-1.762805,1.593035,1.372302>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <-1.762805,1.593035,1.372302>, 0.12, <-1.837538,1.623494,1.576966>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <0.474503,-0.204461,-1.136728>, <-0.271880,0.056324,-1.944072>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <-0.271880,0.056324,-1.944072>, 0.12, <-0.417194,0.107097,-2.101253>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <1.382751,0.959334,0.661704>, <1.560257,1.352618,0.733535>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <1.560257,1.352618,0.733535>, 0.12, <1.649533,1.550417,0.769663>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <0.495015,-1.471871,0.770997>, <0.498362,-1.753395,0.699078>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <0.498362,-1.753395,0.699078>, 0.12, <0.500897,-1.966536,0.644629>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <-1.427521,0.540929,-0.633148>, <1.166343,-0.492275,-0.263408>, 0.04 pigment { color rgb <0.48,0.08,0.72> } finish { emission 0.25 } }
sphere { <-0.130589,0.024327,-0.448278>, 0.075 pigment { color rgb <0.48,0.08,0.72> } }
