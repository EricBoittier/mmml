#version 3.7;
global_settings { assumed_gamma 1.0 }
background { color rgbt <1,1,1,1> }
camera { orthographic location <6.275618,4.365648,-14.734061> look_at <0,0,0> right x*9.386142322800001 up y*7.039606742100001 }
light_source { <-5.457059,10.914119,-10.914119> color rgb <1,1,1> area_light <2,0,0>,<0,2,0>,5,5 adaptive 1 jitter }
light_source { <10.914119,-5.457059,-5.457059> color rgb <0.35,0.40,0.52> }
plane { y, -2.15 pigment { color rgbt <0.91,0.92,0.94,0.82> } finish { diffuse 0.75 } }
cylinder { <3.135170,-1.290859,-0.076325>, <1.482980,-0.572267,0.061856>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <-1.765721,0.686314,-0.056056>, <-2.321890,0.366157,0.734407>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <-1.765721,0.686314,-0.056056>, <-2.259104,0.409171,-0.902640>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <-1.765721,0.686314,-0.056056>, <-1.762932,1.704204,-0.029063>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <1.482980,-0.572267,0.061856>, <1.033429,-0.531488,-0.934348>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <1.482980,-0.572267,0.061856>, <1.575035,0.434593,0.480477>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
cylinder { <1.482980,-0.572267,0.061856>, <0.883031,-1.205825,0.721692>, 0.105 pigment { color rgb <0.55,0.57,0.61> } finish { diffuse 0.7 phong 0.25 } }
sphere { <3.135170,-1.290859,-0.076325>, 0.5304 texture { pigment { color rgb <0.200000,0.720000,0.280000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <-1.765721,0.686314,-0.056056>, 0.3692 texture { pigment { color rgb <0.180000,0.320000,0.880000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <1.482980,-0.572267,0.061856>, 0.3952 texture { pigment { color rgb <0.220000,0.240000,0.280000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <-2.321890,0.366157,0.734407>, 0.2400 texture { pigment { color rgb <0.860000,0.870000,0.900000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <-2.259104,0.409171,-0.902640>, 0.2400 texture { pigment { color rgb <0.860000,0.870000,0.900000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <-1.762932,1.704204,-0.029063>, 0.2400 texture { pigment { color rgb <0.860000,0.870000,0.900000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <1.033429,-0.531488,-0.934348>, 0.2400 texture { pigment { color rgb <0.860000,0.870000,0.900000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <1.575035,0.434593,0.480477>, 0.2400 texture { pigment { color rgb <0.860000,0.870000,0.900000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
sphere { <0.883031,-1.205825,0.721692>, 0.2400 texture { pigment { color rgb <0.860000,0.870000,0.900000> } finish { phong 0.85 phong_size 90 reflection 0.06 } } }
cylinder { <3.409293,-1.440238,0.204642>, <3.786629,-1.645863,0.591399>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <3.786629,-1.645863,0.591399>, 0.12, <3.930217,-1.724109,0.738573>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <-1.420443,0.566532,-0.263023>, <-1.181751,0.483727,-0.406099>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <-1.181751,0.483727,-0.406099>, 0.12, <-1.000891,0.420984,-0.514510>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <1.092908,-0.421031,0.024832>, <0.511732,-0.195701,-0.030331>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <0.511732,-0.195701,-0.030331>, 0.12, <0.307408,-0.116482,-0.049725>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <-2.203306,-0.027597,0.648988>, <-2.102413,-0.362610,0.576312>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <-2.102413,-0.362610,0.576312>, 0.12, <-2.040297,-0.568862,0.531569>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <-1.850691,0.474656,-0.975516>, <-1.393996,0.547882,-1.057008>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <-1.393996,0.547882,-1.057008>, 0.12, <-1.180065,0.582184,-1.095182>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <-1.905603,1.762354,0.361659>, <-2.200449,1.882526,1.169128>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <-2.200449,1.882526,1.169128>, 0.12, <-2.275181,1.912985,1.373791>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <0.756012,-0.434559,-1.234423>, <0.009629,-0.173774,-2.041766>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <0.009629,-0.173774,-2.041766>, 0.12, <-0.135684,-0.123002,-2.198948>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <1.745471,0.812210,0.549448>, <1.922977,1.205493,0.621279>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <1.922977,1.205493,0.621279>, 0.12, <2.012253,1.403293,0.657406>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <0.887870,-1.612729,0.617744>, <0.891217,-1.894253,0.545825>, 0.045 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cone { <0.891217,-1.894253,0.545825>, 0.12, <0.893751,-2.107393,0.491376>, 0 pigment { color rgb <0.78,0.04,0.12> } finish { emission 0.18 } }
cylinder { <-1.765721,0.686314,-0.406056>, <1.482980,-0.572267,-0.288144>, 0.04 pigment { color rgb <0.48,0.08,0.72> } finish { emission 0.25 } }
sphere { <-0.141370,0.057024,-0.347100>, 0.075 pigment { color rgb <0.48,0.08,0.72> } }
