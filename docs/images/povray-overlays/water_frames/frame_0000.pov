#version 3.7;
global_settings { assumed_gamma 1.0 }
background { color rgbt <1,1,1,1> }
camera { orthographic location <5.186297,3.607859,-12.176523> look_at <0,0,0> right x*7.756895963781742 up y*5.8176719728363055 }
light_source { <-4.509823,9.019646,-9.019646> color rgb <1,1,1> area_light <2,0,0>,<0,2,0>,5,5 adaptive 1 jitter }
light_source { <9.019646,-4.509823,-4.509823> color rgb <0.35,0.40,0.52> }
plane { y, -2.15 pigment { color rgbt <0.91,0.92,0.94,0.82> } finish { diffuse 0.75 } }
cylinder { <-1.680075,-2.172015,-1.955460>, <-2.116299,-1.379965,-2.096830>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <-1.680075,-2.172015,-1.955460>, <-1.060317,-1.545812,-2.199058>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <1.897834,1.570162,-1.600027>, <2.039779,2.299943,-1.047724>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <1.897834,1.570162,-1.600027>, <1.506023,2.316659,-1.973601>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <2.094884,-1.604762,1.886170>, <1.875680,-2.092021,1.140909>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <2.094884,-1.604762,1.886170>, <1.299741,-2.027957,2.039444>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <-1.474585,1.563302,1.762826>, <-2.113042,1.976130,2.310765>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <-1.474585,1.563302,1.762826>, <-2.269624,1.096336,1.732586>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
sphere { <-1.680075,-2.172015,-1.955460>, 0.3432 texture { pigment { color rgb <0.860000,0.160000,0.130000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <-2.116299,-1.379965,-2.096830>, 0.2400 texture { pigment { color rgb <0.860000,0.870000,0.900000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <-1.060317,-1.545812,-2.199058>, 0.2400 texture { pigment { color rgb <0.860000,0.870000,0.900000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <1.897834,1.570162,-1.600027>, 0.3432 texture { pigment { color rgb <0.860000,0.160000,0.130000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <2.039779,2.299943,-1.047724>, 0.2400 texture { pigment { color rgb <0.860000,0.870000,0.900000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <1.506023,2.316659,-1.973601>, 0.2400 texture { pigment { color rgb <0.860000,0.870000,0.900000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <2.094884,-1.604762,1.886170>, 0.3432 texture { pigment { color rgb <0.860000,0.160000,0.130000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <1.875680,-2.092021,1.140909>, 0.2400 texture { pigment { color rgb <0.860000,0.870000,0.900000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <1.299741,-2.027957,2.039444>, 0.2400 texture { pigment { color rgb <0.860000,0.870000,0.900000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <-1.474585,1.563302,1.762826>, 0.3432 texture { pigment { color rgb <0.860000,0.160000,0.130000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <-2.113042,1.976130,2.310765>, 0.2400 texture { pigment { color rgb <0.860000,0.870000,0.900000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <-2.269624,1.096336,1.732586>, 0.2400 texture { pigment { color rgb <0.860000,0.870000,0.900000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
cylinder { <-1.527577,-1.789346,-2.037370>, <-1.125251,-0.779771,-2.253469>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <-1.125251,-0.779771,-2.253469>, 0.12, <-1.045372,-0.579326,-2.296375>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <-1.862852,-1.711699,-2.050816>, <-1.617656,-2.032632,-2.006300>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <-1.617656,-2.032632,-2.006300>, 0.12, <-1.484898,-2.206398,-1.982198>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <-1.401547,-1.772548,-2.106578>, <-2.174168,-2.285929,-1.897183>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <-2.174168,-2.285929,-1.897183>, 0.12, <-2.352907,-2.404696,-1.848741>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <1.922947,1.159187,-1.682909>, <1.990512,0.053471,-1.905900>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <1.990512,0.053471,-1.905900>, 0.12, <2.003667,-0.161802,-1.949314>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <2.152103,2.565913,-0.742695>, <2.403860,3.162044,-0.059018>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <2.403860,3.162044,-0.059018>, 0.12, <2.462697,3.301361,0.100759>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <1.317224,2.603619,-2.215280>, <1.041823,3.022207,-2.567817>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <1.041823,3.022207,-2.567817>, 0.12, <0.942928,3.172519,-2.694411>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <2.355799,-1.279980,1.939462>, <2.652945,-0.910100,2.000154>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <2.652945,-0.910100,2.000154>, 0.12, <2.789614,-0.739976,2.028069>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <1.884985,-2.328608,0.794009>, <1.889729,-2.449247,0.617119>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <1.889729,-2.449247,0.617119>, 0.12, <1.894603,-2.573174,0.435410>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <1.034883,-2.286505,2.237946>, <0.826644,-2.489784,2.394013>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <0.826644,-2.489784,2.394013>, 0.12, <0.687909,-2.625214,2.497991>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <-1.074920,1.579200,1.634707>, <-0.726917,1.593044,1.523149>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <-0.726917,1.593044,1.523149>, 0.12, <-0.517569,1.601372,1.456038>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <-2.383311,2.253959,2.472523>, <-2.493454,2.367184,2.538446>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <-2.493454,2.367184,2.538446>, 0.12, <-2.635024,2.512713,2.623176>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <-2.565759,0.821627,1.617527>, <-2.738555,0.661333,1.550390>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <-2.738555,0.661333,1.550390>, 0.12, <-2.893673,0.517438,1.490122>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
