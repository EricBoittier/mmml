#version 3.7;
global_settings { assumed_gamma 1.0 }
background { color rgbt <1,1,1,1> }
camera { orthographic location <3.450000,2.400000,-8.100000> look_at <0,0,0> right x*6.1499999999999995 up y*4.612499999999999 }
light_source { <-3.000000,6.000000,-6.000000> color rgb <1,1,1> area_light <2,0,0>,<0,2,0>,5,5 adaptive 1 jitter }
light_source { <6.000000,-3.000000,-3.000000> color rgb <0.35,0.40,0.52> }
plane { y, -2.15 pigment { color rgbt <0.91,0.92,0.94,0.82> } finish { diffuse 0.75 } }
cylinder { <0.075743,-0.097204,0.063533>, <0.545333,-1.026584,-0.242766>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <0.075743,-0.097204,0.063533>, <0.043548,0.034130,1.131150>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <0.075743,-0.097204,0.063533>, <-1.612631,-0.140386,-0.434430>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <0.075743,-0.097204,0.063533>, <0.948007,1.230045,-0.517486>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
sphere { <0.075743,-0.097204,0.063533>, 0.3952 texture { pigment { color rgb <0.220000,0.240000,0.280000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <0.545333,-1.026584,-0.242766>, 0.2400 texture { pigment { color rgb <0.860000,0.870000,0.900000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <0.043548,0.034130,1.131150>, 0.2400 texture { pigment { color rgb <0.860000,0.870000,0.900000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <-1.612631,-0.140386,-0.434430>, 0.5304 texture { pigment { color rgb <0.200000,0.720000,0.280000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <0.948007,1.230045,-0.517486>, 0.5304 texture { pigment { color rgb <0.200000,0.720000,0.280000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <0.075743,-0.097204,0.063533>, 0.4751 pigment { color rgbt <0.88,0.12,0.2,0.78> } finish { emission 0.05 phong 0.35 } no_shadow }
sphere { <0.545333,-1.026584,-0.242766>, 0.3595 pigment { color rgbt <0.16,0.38,0.92,0.78> } finish { emission 0.05 phong 0.35 } no_shadow }
sphere { <0.043548,0.034130,1.131150>, 0.3595 pigment { color rgbt <0.16,0.38,0.92,0.78> } finish { emission 0.05 phong 0.35 } no_shadow }
sphere { <-1.612631,-0.140386,-0.434430>, 0.6450 pigment { color rgbt <0.88,0.12,0.2,0.78> } finish { emission 0.05 phong 0.35 } no_shadow }
sphere { <0.948007,1.230045,-0.517486>, 0.6450 pigment { color rgbt <0.88,0.12,0.2,0.78> } finish { emission 0.05 phong 0.35 } no_shadow }
cylinder { <0.000000,0.000000,0.000000>, <1.047550,-1.252632,1.206007>, 0.065 pigment { color rgb <0.950000,0.620000,0.060000> } finish { emission 0.16 phong 0.5 } }
cone { <1.047550,-1.252632,1.206007>, 0.182, <1.161078,-1.388385,1.336707>, 0 pigment { color rgb <0.950000,0.620000,0.060000> } finish { emission 0.16 phong 0.5 } }
cylinder { <-0.176734,-0.093445,0.005745>, <-0.571206,-0.302014,0.018567>, 0.018 pigment { color rgb <0.050000,0.620000,0.620000> } finish { emission 0.16 phong 0.5 } }
cone { <-0.571206,-0.302014,0.018567>, 0.05039999999999999, <-0.724611,-0.383124,0.023553>, 0 pigment { color rgb <0.050000,0.620000,0.620000> } finish { emission 0.16 phong 0.5 } }
cylinder { <0.176734,0.093445,-0.005745>, <0.571206,0.302014,-0.018567>, 0.018 pigment { color rgb <0.050000,0.620000,0.620000> } finish { emission 0.16 phong 0.5 } }
cone { <0.571206,0.302014,-0.018567>, 0.05039999999999999, <0.724611,0.383124,-0.023553>, 0 pigment { color rgb <0.050000,0.620000,0.620000> } finish { emission 0.16 phong 0.5 } }
cylinder { <-0.075134,0.134245,-0.127801>, <-0.179484,0.320693,-0.305298>, 0.018 pigment { color rgb <0.480000,0.200000,0.780000> } finish { emission 0.16 phong 0.5 } }
cone { <-0.179484,0.320693,-0.305298>, 0.05039999999999999, <-0.220064,0.393201,-0.374325>, 0 pigment { color rgb <0.480000,0.200000,0.780000> } finish { emission 0.16 phong 0.5 } }
cylinder { <0.075134,-0.134245,0.127801>, <0.179484,-0.320693,0.305298>, 0.018 pigment { color rgb <0.480000,0.200000,0.780000> } finish { emission 0.16 phong 0.5 } }
cone { <0.179484,-0.320693,0.305298>, 0.05039999999999999, <0.220064,-0.393201,0.374325>, 0 pigment { color rgb <0.480000,0.200000,0.780000> } finish { emission 0.16 phong 0.5 } }
cylinder { <-0.055856,0.115092,0.153733>, <-0.167296,0.344718,0.460453>, 0.018 pigment { color rgb <0.480000,0.200000,0.780000> } finish { emission 0.16 phong 0.5 } }
cone { <-0.167296,0.344718,0.460453>, 0.05039999999999999, <-0.210633,0.434016,0.579733>, 0 pigment { color rgb <0.480000,0.200000,0.780000> } finish { emission 0.16 phong 0.5 } }
cylinder { <0.055856,-0.115092,-0.153733>, <0.167296,-0.344718,-0.460453>, 0.018 pigment { color rgb <0.480000,0.200000,0.780000> } finish { emission 0.16 phong 0.5 } }
cone { <0.167296,-0.344718,-0.460453>, 0.05039999999999999, <0.210633,-0.434016,-0.579733>, 0 pigment { color rgb <0.480000,0.200000,0.780000> } finish { emission 0.16 phong 0.5 } }
