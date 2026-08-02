#version 3.7;
global_settings { assumed_gamma 1.0 }
background { color rgbt <1,1,1,1> }
camera { orthographic location <6.210000,4.320000,-14.580000> look_at <0,0,0> right x*7.83 up y*4.586142857142858 }
light_source { <-5.400000,10.800000,-10.800000> color rgb <1,1,1> area_light <2,0,0>,<0,2,0>,5,5 adaptive 1 jitter }
light_source { <10.800000,-5.400000,-5.400000> color rgb <0.35,0.40,0.52> }
plane { y, -2.15 pigment { color rgbt <0.91,0.92,0.94,0.82> } finish { diffuse 0.75 } }
cylinder { <-2.700000,0.000000,0.397539>, <-2.700000,0.763239,-0.198770>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <-2.700000,0.000000,0.397539>, <-2.700000,-0.763239,-0.198770>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <2.700000,0.000000,0.397539>, <2.700000,0.763239,-0.198770>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <2.700000,0.000000,0.397539>, <2.700000,-0.763239,-0.198770>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
sphere { <-2.700000,0.000000,0.397539>, 0.3432 texture { pigment { color rgb <0.860000,0.160000,0.130000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <-2.700000,0.763239,-0.198770>, 0.2400 texture { pigment { color rgb <0.860000,0.870000,0.900000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <-2.700000,-0.763239,-0.198770>, 0.2400 texture { pigment { color rgb <0.860000,0.870000,0.900000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <2.700000,0.000000,0.397539>, 0.3432 texture { pigment { color rgb <0.860000,0.160000,0.130000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <2.700000,0.763239,-0.198770>, 0.2400 texture { pigment { color rgb <0.860000,0.870000,0.900000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <2.700000,-0.763239,-0.198770>, 0.2400 texture { pigment { color rgb <0.860000,0.870000,0.900000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
cylinder { <-2.700000,0.000000,0.397539>, <-2.500000,0.000000,0.597539>, 0.018 pigment { color rgbt <.25,.28,.34,.45> } no_shadow }
sphere { <-2.500000,0.000000,0.597539>, 0.184 pigment { color rgb <0.880000,0.100000,0.180000> } finish { emission .12 phong .7 } no_shadow }
cylinder { <-2.700000,0.000000,0.397539>, <-2.900000,0.000000,0.597539>, 0.018 pigment { color rgbt <.25,.28,.34,.45> } no_shadow }
sphere { <-2.900000,0.000000,0.597539>, 0.184 pigment { color rgb <0.880000,0.100000,0.180000> } finish { emission .12 phong .7 } no_shadow }
cylinder { <-2.700000,0.763239,-0.198770>, <-2.600000,0.883239,-0.198770>, 0.018 pigment { color rgbt <.25,.28,.34,.45> } no_shadow }
sphere { <-2.600000,0.883239,-0.198770>, 0.152 pigment { color rgb <0.160000,0.380000,0.920000> } finish { emission .12 phong .7 } no_shadow }
cylinder { <-2.700000,0.763239,-0.198770>, <-2.800000,0.643239,-0.198770>, 0.018 pigment { color rgbt <.25,.28,.34,.45> } no_shadow }
sphere { <-2.800000,0.643239,-0.198770>, 0.152 pigment { color rgb <0.160000,0.380000,0.920000> } finish { emission .12 phong .7 } no_shadow }
cylinder { <-2.700000,-0.763239,-0.198770>, <-2.600000,-0.643239,-0.198770>, 0.018 pigment { color rgbt <.25,.28,.34,.45> } no_shadow }
sphere { <-2.600000,-0.643239,-0.198770>, 0.152 pigment { color rgb <0.160000,0.380000,0.920000> } finish { emission .12 phong .7 } no_shadow }
cylinder { <-2.700000,-0.763239,-0.198770>, <-2.800000,-0.883239,-0.198770>, 0.018 pigment { color rgbt <.25,.28,.34,.45> } no_shadow }
sphere { <-2.800000,-0.883239,-0.198770>, 0.152 pigment { color rgb <0.160000,0.380000,0.920000> } finish { emission .12 phong .7 } no_shadow }
cylinder { <2.700000,0.000000,0.000000>, <2.700000,-0.000000,-1.230000>, 0.055 pigment { color rgb <0.950000,0.620000,0.060000> } finish { emission 0.16 phong 0.5 } }
cone { <2.700000,-0.000000,-1.230000>, 0.154, <2.700000,-0.000000,-1.450000>, 0 pigment { color rgb <0.950000,0.620000,0.060000> } finish { emission 0.16 phong 0.5 } }
cylinder { <2.700000,0.000000,-0.180000>, <2.700000,0.000000,-0.540000>, 0.017 pigment { color rgb <0.050000,0.620000,0.620000> } finish { emission 0.16 phong 0.5 } }
cone { <2.700000,0.000000,-0.540000>, 0.0476, <2.700000,0.000000,-0.680000>, 0 pigment { color rgb <0.050000,0.620000,0.620000> } finish { emission 0.16 phong 0.5 } }
cylinder { <2.700000,-0.000000,0.180000>, <2.700000,-0.000000,0.540000>, 0.017 pigment { color rgb <0.050000,0.620000,0.620000> } finish { emission 0.16 phong 0.5 } }
cone { <2.700000,-0.000000,0.540000>, 0.0476, <2.700000,-0.000000,0.680000>, 0 pigment { color rgb <0.050000,0.620000,0.620000> } finish { emission 0.16 phong 0.5 } }
cylinder { <2.879967,-0.003443,-0.000000>, <3.239901,-0.010330,-0.000000>, 0.017 pigment { color rgb <0.050000,0.620000,0.620000> } finish { emission 0.16 phong 0.5 } }
cone { <3.239901,-0.010330,-0.000000>, 0.0476, <3.379876,-0.013009,-0.000000>, 0 pigment { color rgb <0.050000,0.620000,0.620000> } finish { emission 0.16 phong 0.5 } }
cylinder { <2.520033,0.003443,0.000000>, <2.160099,0.010330,0.000000>, 0.017 pigment { color rgb <0.050000,0.620000,0.620000> } finish { emission 0.16 phong 0.5 } }
cone { <2.160099,0.010330,0.000000>, 0.0476, <2.020124,0.013009,0.000000>, 0 pigment { color rgb <0.050000,0.620000,0.620000> } finish { emission 0.16 phong 0.5 } }
cylinder { <2.703443,0.179967,0.000000>, <2.710330,0.539901,0.000000>, 0.017 pigment { color rgb <0.480000,0.200000,0.780000> } finish { emission 0.16 phong 0.5 } }
cone { <2.710330,0.539901,0.000000>, 0.0476, <2.713009,0.679876,0.000000>, 0 pigment { color rgb <0.480000,0.200000,0.780000> } finish { emission 0.16 phong 0.5 } }
cylinder { <2.696557,-0.179967,0.000000>, <2.689670,-0.539901,0.000000>, 0.017 pigment { color rgb <0.480000,0.200000,0.780000> } finish { emission 0.16 phong 0.5 } }
cone { <2.689670,-0.539901,0.000000>, 0.0476, <2.686991,-0.679876,0.000000>, 0 pigment { color rgb <0.480000,0.200000,0.780000> } finish { emission 0.16 phong 0.5 } }
