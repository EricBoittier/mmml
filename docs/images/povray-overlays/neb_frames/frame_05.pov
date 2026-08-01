#version 3.7;
global_settings { assumed_gamma 1.0 }
background { color rgbt <1,1,1,1> }
camera { orthographic location <5.393654,3.752107,-12.663362> look_at <0,0,0> right x*8.067030376800002 up y*6.050272782600002 }
light_source { <-4.690134,9.380268,-9.380268> color rgb <1,1,1> area_light <2,0,0>,<0,2,0>,5,5 adaptive 1 jitter }
light_source { <9.380268,-4.690134,-4.690134> color rgb <0.35,0.40,0.52> }
plane { y, -2.15 pigment { color rgbt <0.91,0.92,0.94,0.82> } finish { diffuse 0.75 } }
cylinder { <-1.066681,0.510357,0.097774>, <0.384171,-0.219610,0.031081>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <-1.066681,0.510357,0.097774>, <-1.291812,0.874323,1.029101>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <-1.066681,0.510357,0.097774>, <-1.832145,-0.112698,-0.178171>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <-1.066681,0.510357,0.097774>, <-1.071170,1.303619,-0.546600>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <0.384171,-0.219610,0.031081>, <0.220532,-1.245362,-0.307183>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <0.384171,-0.219610,0.031081>, <0.982958,0.336191,-0.689060>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <0.384171,-0.219610,0.031081>, <0.816158,-0.159433,1.028331>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
sphere { <2.857989,-1.287386,-0.465272>, 0.5304 texture { pigment { color rgb <0.200000,0.720000,0.280000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <-1.066681,0.510357,0.097774>, 0.3692 texture { pigment { color rgb <0.180000,0.320000,0.880000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <0.384171,-0.219610,0.031081>, 0.3952 texture { pigment { color rgb <0.220000,0.240000,0.280000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <-1.291812,0.874323,1.029101>, 0.2400 texture { pigment { color rgb <0.860000,0.870000,0.900000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <-1.832145,-0.112698,-0.178171>, 0.2400 texture { pigment { color rgb <0.860000,0.870000,0.900000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <-1.071170,1.303619,-0.546600>, 0.2400 texture { pigment { color rgb <0.860000,0.870000,0.900000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <0.220532,-1.245362,-0.307183>, 0.2400 texture { pigment { color rgb <0.860000,0.870000,0.900000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <0.982958,0.336191,-0.689060>, 0.2400 texture { pigment { color rgb <0.860000,0.870000,0.900000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <0.816158,-0.159433,1.028331>, 0.2400 texture { pigment { color rgb <0.860000,0.870000,0.900000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
cylinder { <3.132112,-1.436766,-0.184304>, <3.509448,-1.642390,0.202453>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <3.509448,-1.642390,0.202453>, 0.12, <3.653036,-1.720636,0.349626>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <-0.721403,0.390575,-0.109193>, <-0.482711,0.307769,-0.252269>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <-0.482711,0.307769,-0.252269>, 0.12, <-0.301851,0.245026,-0.360680>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <-0.005902,-0.068374,-0.005944>, <-0.587077,0.156955,-0.061107>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <-0.587077,0.156955,-0.061107>, 0.12, <-0.791401,0.236174,-0.080501>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <-1.173229,0.480570,0.943682>, <-1.072335,0.145556,0.871005>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <-1.072335,0.145556,0.871005>, 0.12, <-1.010220,-0.060695,0.826262>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <-1.423732,-0.047213,-0.251047>, <-0.967037,0.026014,-0.332539>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <-0.967037,0.026014,-0.332539>, 0.12, <-0.753107,0.060315,-0.370712>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <-1.213841,1.361768,-0.155879>, <-1.508687,1.481941,0.651590>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <-1.508687,1.481941,0.651590>, 0.12, <-1.583419,1.512400,0.856254>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <-0.056885,-1.148433,-0.607258>, <-0.803268,-0.887648,-1.414601>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <-0.803268,-0.887648,-1.414601>, 0.12, <-0.948581,-0.836876,-1.571783>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <1.153394,0.713808,-0.620090>, <1.330900,1.107091,-0.548258>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <1.330900,1.107091,-0.548258>, 0.12, <1.420175,1.304891,-0.512131>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <0.820996,-0.566338,0.924382>, <0.824344,-0.847861,0.852464>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <0.824344,-0.847861,0.852464>, 0.12, <0.826878,-1.061002,0.798015>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <-1.066681,0.510357,-0.252226>, <0.384171,-0.219610,-0.318919>, 0.04 pigment { color rgb <0.48,0.08,0.72> } finish { emission 0.25 } }
sphere { <-0.341255,0.145373,-0.285573>, 0.075 pigment { color rgb <0.48,0.08,0.72> } }
