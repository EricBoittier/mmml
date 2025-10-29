#version 3.6;
#include "colors.inc"
#include "finish.inc"

global_settings {assumed_gamma 2.2 max_trace_level 6}
background {color White transmit 1.0}
camera {orthographic
  right -7.78*x up 8.06*y
  direction 1.00*z
  location <0,0,50.00> look_at <0,0,0>}


light_source {<  2.00,   3.00,  40.00> color White
  area_light <0.70, 0, 0>, <0, 0.70, 0>, 3, 3
  adaptive 1 jitter}
// no fog
#declare simple = finish {phong 0.7 ambient 0.4 diffuse 0.55}
#declare pale = finish {ambient 0.9 diffuse 0.30 roughness 0.001 specular 0.2 }
#declare intermediate = finish {ambient 0.4 diffuse 0.6 specular 0.1 roughness 0.04}
#declare vmd = finish {ambient 0.2 diffuse 0.80 phong 0.25 phong_size 10.0 specular 0.2 roughness 0.1}
#declare jmol = finish {ambient 0.4 diffuse 0.6 specular 1 roughness 0.001 metallic}
#declare ase2 = finish {ambient 0.2 brilliance 3 diffuse 0.6 metallic specular 0.7 roughness 0.04 reflection 0.15}
#declare ase3 = finish {ambient 0.4 brilliance 2 diffuse 0.6 metallic specular 1.0 roughness 0.001 reflection 0.0}
#declare glass = finish {ambient 0.4 diffuse 0.35 specular 1.0 roughness 0.001}
#declare glass2 = finish {ambient 0.3 diffuse 0.3 specular 1.0 reflection 0.25 roughness 0.001}
#declare Rcell = 0.070;
#declare Rbond = 0.100;

#macro atom(LOC, R, COL, TRANS, FIN)
  sphere{LOC, R texture{pigment{color COL transmit TRANS} finish{FIN}}}
#end
#macro constrain(LOC, R, COL, TRANS FIN)
union{torus{R, Rcell rotate 45*z texture{pigment{color COL transmit TRANS} finish{FIN}}}
     torus{R, Rcell rotate -45*z texture{pigment{color COL transmit TRANS} finish{FIN}}}
     translate LOC}
#end

// no cell vertices
atom(< -1.22,   0.47,  -0.24>, 0.68, rgb <0.56, 0.56, 0.56>, 0.0, ase2) // #0
atom(< -1.20,  -0.92,  -0.37>, 0.68, rgb <0.56, 0.56, 0.56>, 0.0, ase2) // #1
atom(<  1.21,  -0.93,  -0.25>, 0.68, rgb <0.56, 0.56, 0.56>, 0.0, ase2) // #2
atom(< -2.15,  -1.46,  -0.46>, 0.28, rgb <1.00, 1.00, 1.00>, 0.0, ase2) // #3
atom(<  1.21,   0.46,  -0.12>, 0.68, rgb <0.56, 0.56, 0.56>, 0.0, ase2) // #4
atom(< -0.01,   2.47,   0.00>, 0.51, rgb <0.56, 0.87, 0.31>, 0.0, ase2) // #5
atom(< -0.00,   1.13,  -0.12>, 0.68, rgb <0.56, 0.56, 0.56>, 0.0, ase2) // #6
atom(<  0.01,  -2.71,  -0.47>, 0.28, rgb <1.00, 1.00, 1.00>, 0.0, ase2) // #7
atom(< -2.15,   1.04,  -0.24>, 0.28, rgb <1.00, 1.00, 1.00>, 0.0, ase2) // #8
atom(<  0.00,  -1.62,  -0.37>, 0.68, rgb <0.56, 0.56, 0.56>, 0.0, ase2) // #9
atom(<  2.14,   1.03,  -0.02>, 0.28, rgb <1.00, 1.00, 1.00>, 0.0, ase2) // #10
atom(<  2.15,  -1.47,  -0.25>, 0.28, rgb <1.00, 1.00, 1.00>, 0.0, ase2) // #11

// no constraints
