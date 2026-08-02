#version 3.7;
global_settings { assumed_gamma 1.0 }
background { color rgbt <1,1,1,1> }
camera { orthographic location <6.051723,4.209894,-14.208393> look_at <0,0,0> right x*9.051272581656372 up y*6.788454436242279 }
light_source { <-5.262368,10.524736,-10.524736> color rgb <1,1,1> area_light <2,0,0>,<0,2,0>,5,5 adaptive 1 jitter }
light_source { <10.524736,-5.262368,-5.262368> color rgb <0.35,0.40,0.52> }
plane { y, -2.15 pigment { color rgbt <0.91,0.92,0.94,0.82> } finish { diffuse 0.75 } }
cylinder { <-1.438607,-1.949178,-1.569859>, <-1.784113,-2.720850,-1.977590>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <-1.438607,-1.949178,-1.569859>, <-2.057637,-1.745092,-2.227019>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <1.387976,1.621257,-1.555403>, <1.426590,2.541517,-1.355306>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <1.387976,1.621257,-1.555403>, <1.448915,2.080727,-2.356029>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <2.138238,-0.830353,1.602865>, <2.582649,-1.510242,2.058626>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <2.138238,-0.830353,1.602865>, <2.402233,-1.506537,0.999548>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <-1.684055,1.499106,1.798671>, <-2.595189,1.432189,1.977638>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <-1.684055,1.499106,1.798671>, <-1.827001,1.087456,2.603858>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
sphere { <-1.438607,-1.949178,-1.569859>, 0.3432 texture { pigment { color rgb <0.860000,0.160000,0.130000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <-1.784113,-2.720850,-1.977590>, 0.2400 texture { pigment { color rgb <0.860000,0.870000,0.900000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <-2.057637,-1.745092,-2.227019>, 0.2400 texture { pigment { color rgb <0.860000,0.870000,0.900000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <1.387976,1.621257,-1.555403>, 0.3432 texture { pigment { color rgb <0.860000,0.160000,0.130000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <1.426590,2.541517,-1.355306>, 0.2400 texture { pigment { color rgb <0.860000,0.870000,0.900000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <1.448915,2.080727,-2.356029>, 0.2400 texture { pigment { color rgb <0.860000,0.870000,0.900000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <2.138238,-0.830353,1.602865>, 0.3432 texture { pigment { color rgb <0.860000,0.160000,0.130000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <2.582649,-1.510242,2.058626>, 0.2400 texture { pigment { color rgb <0.860000,0.870000,0.900000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <2.402233,-1.506537,0.999548>, 0.2400 texture { pigment { color rgb <0.860000,0.870000,0.900000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <-1.684055,1.499106,1.798671>, 0.3432 texture { pigment { color rgb <0.860000,0.160000,0.130000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <-2.595189,1.432189,1.977638>, 0.2400 texture { pigment { color rgb <0.860000,0.870000,0.900000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <-1.827001,1.087456,2.603858>, 0.2400 texture { pigment { color rgb <0.860000,0.870000,0.900000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
cylinder { <-1.700677,-2.081863,-1.870048>, <-2.334961,-2.402998,-2.596590>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <-2.334961,-2.402998,-2.596590>, 0.12, <-2.472236,-2.472499,-2.753832>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <-1.591841,-2.451884,-1.718574>, <-1.394857,-2.176326,-1.453210>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <-1.394857,-2.176326,-1.453210>, 0.12, <-1.294143,-2.035439,-1.317535>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <-1.767369,-1.766728,-1.924237>, <-1.426010,-1.792173,-1.568161>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <-1.426010,-1.792173,-1.568161>, 0.12, <-1.273965,-1.803506,-1.409562>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <1.519818,1.937791,-1.797939>, <1.480968,1.844517,-1.726470>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <1.480968,1.844517,-1.726470>, 0.12, <1.550028,2.010320,-1.853513>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <1.419538,2.355992,-1.732043>, <1.416189,2.267884,-1.910960>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <1.416189,2.267884,-1.910960>, 0.12, <1.412495,2.170704,-2.108299>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <1.457305,2.296616,-1.995861>, <1.460850,2.387831,-1.843686>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <1.460850,2.387831,-1.843686>, 0.12, <1.465244,2.500916,-1.655027>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <2.327144,-1.205450,1.606973>, <2.835390,-2.214640,1.618026>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <2.835390,-2.214640,1.618026>, 0.12, <2.934341,-2.411119,1.620177>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <2.388568,-1.225884,1.818059>, <2.034733,-0.707462,1.379476>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <2.034733,-0.707462,1.379476>, 0.12, <1.933071,-0.558512,1.253464>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <2.318995,-1.253590,1.324340>, <2.215536,-0.939197,1.728031>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <2.215536,-0.939197,1.728031>, 0.12, <2.171935,-0.806701,1.898160>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <-1.512316,1.674556,1.457902>, <-1.163179,2.031240,0.765132>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <-1.163179,2.031240,0.765132>, 0.12, <-1.073220,2.123142,0.586634>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <-3.003455,1.503798,1.909880>, <-3.341636,1.563114,1.853753>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <-3.341636,1.563114,1.853753>, 0.12, <-3.555490,1.600623,1.818261>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <-1.805668,0.889633,2.973737>, <-1.759194,0.458679,3.779512>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <-1.759194,0.458679,3.779512>, 0.12, <-1.748019,0.355057,3.973259>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
