#version 3.6;
#include "colors.inc"
#include "finish.inc"

global_settings {assumed_gamma 2.2 max_trace_level 6}
background {color White}
camera {orthographic
  right -3.30*x up 0.84*y
  direction 1.00*z
  location <0,0,60.00> look_at <0,0,0>}


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
#declare Rcell = 0.000;
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
atom(<  0.00,   0.00,  -0.00>, 0.40, rgb <0.56, 0.56, 0.56>, 0.0, jmol) // #0
atom(<  1.17,   0.00,   0.00>, 0.40, rgb <1.00, 0.05, 0.05>, 0.0, jmol) // #1
atom(< -1.17,   0.00,  -0.00>, 0.40, rgb <1.00, 0.05, 0.05>, 0.0, jmol) // #2
cylinder {<  0.00,   0.00,  -0.00>, <  0.58,   0.00,  -0.00>, Rbond texture{pigment {color rgb <0.56, 0.56, 0.56> transmit 0.0} finish{jmol}}}
cylinder {<  1.17,   0.00,   0.00>, <  0.58,   0.00,  -0.00>, Rbond texture{pigment {color rgb <1.00, 0.05, 0.05> transmit 0.0} finish{jmol}}}
cylinder {<  0.00,   0.00,  -0.00>, < -0.58,   0.00,  -0.00>, Rbond texture{pigment {color rgb <0.56, 0.56, 0.56> transmit 0.0} finish{jmol}}}
cylinder {< -1.17,   0.00,  -0.00>, < -0.58,   0.00,  -0.00>, Rbond texture{pigment {color rgb <1.00, 0.05, 0.05> transmit 0.0} finish{jmol}}}
// no constraints


// Enhanced rendering settings
// Added by ase_povray_viz.py


// Soft area lighting for better appearance
#declare light_color = rgb <1.0, 1.0, 1.0>;

// Main key light (soft area light)
light_source {
    <20, 30, 40>
    color light_color * 0.8
    area_light <8, 0, 0>, <0, 8, 0>, 5, 5
    adaptive 1
    jitter
    fade_distance 60
    fade_power 1
}

// Fill light (softer, from opposite side)
light_source {
    <-15, 20, -30>
    color light_color * 0.4
    area_light <6, 0, 0>, <0, 6, 0>, 4, 4
    adaptive 1
    jitter
    fade_distance 50
    fade_power 1
}

// Back rim light (subtle)
light_source {
    <-10, -15, -40>
    color light_color * 0.3
    area_light <5, 0, 0>, <0, 5, 0>, 3, 3
    adaptive 1
    jitter
}

// Ambient illumination
global_settings {
    ambient_light rgb <0.15, 0.15, 0.15>
    max_trace_level 15
}

// Enhanced charge texture with anisotropic properties
#declare charge_finish = finish {
    ambient 0.2
    diffuse 0.65
    specular 0.5
    roughness 0.005
    metallic 0.15
    phong 0.6
    phong_size 80
    brilliance 1.2
}

// Enhanced ESP surface finish
#declare esp_finish = finish {
    ambient 0.25
    diffuse 0.6
    specular 0.3
    roughness 0.02
    brilliance 1.0
}

