#version 3.6;
#include "colors.inc"
#include "finish.inc"

global_settings {assumed_gamma 2.2 max_trace_level 6}
background {color White}
camera {orthographic
  right -2.05*x up 2.05*y
  direction 1.00*z
  location <0,0,300.00> look_at <0,0,0>}


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
atom(<  0.00,   0.00,  -0.00>, 0.15, rgb <0.56, 0.56, 0.56>, 0.0, jmol) // #0
atom(<  0.83,  -0.83,   0.00>, 0.15, rgb <1.00, 0.05, 0.05>, 0.0, jmol) // #1
atom(< -0.83,   0.83,  -0.00>, 0.15, rgb <1.00, 0.05, 0.05>, 0.0, jmol) // #2
cylinder {<  0.00,   0.00,  -0.00>, <  0.41,  -0.41,  -0.00>, Rbond texture{pigment {color rgb <0.56, 0.56, 0.56> transmit 0.980} finish{jmol}}}
cylinder {<  0.83,  -0.83,   0.00>, <  0.41,  -0.41,  -0.00>, Rbond texture{pigment {color rgb <1.00, 0.05, 0.05> transmit 0.980} finish{jmol}}}
cylinder {<  0.00,   0.00,  -0.00>, < -0.41,   0.41,  -0.00>, Rbond texture{pigment {color rgb <0.56, 0.56, 0.56> transmit 0.980} finish{jmol}}}
cylinder {< -0.83,   0.83,  -0.00>, < -0.41,   0.41,  -0.00>, Rbond texture{pigment {color rgb <1.00, 0.05, 0.05> transmit 0.980} finish{jmol}}}
// no constraints


// Enhanced rendering settings
// Added by ase_povray_viz.py


// Soft lighting environment
#declare light_color = rgb <1.0, 1.0, 1.0>;

// Main light (broad, gentle)
light_source {
    <24, 36, 42>
    color light_color * 0.45
    area_light <12, 0, 0>, <0, 12, 0>, 6, 6
    adaptive 1
    jitter
    fade_distance 75
    fade_power 2
}

// Fill light (diffuse bounce)
light_source {
    <-18, 22, -28>
    color light_color * 0.25
    area_light <10, 0, 0>, <0, 10, 0>, 5, 5
    adaptive 1
    jitter
    fade_distance 65
    fade_power 2
}

// Rim light (very soft accent)
light_source {
    <-8, -12, -35>
    color light_color * 0.18
    area_light <6, 0, 0>, <0, 6, 0>, 4, 4
    adaptive 1
    jitter
}

// Ambient illumination
global_settings {
    ambient_light rgb <0.20, 0.20, 0.22>
    max_trace_level 15
}

// Sky tint for subtle background colour
sky_sphere {
    pigment {
        gradient y
        color_map {
            [0.0 color rgb <0.92, 0.95, 0.98>]
            [1.0 color rgb <1.0, 1.0, 1.0>]
        }
    }
}

// Soft grid plane for scale reference
#declare grid_color_light = rgb <0.94, 0.96, 0.98>;
#declare grid_color_dark  = rgb <0.87, 0.90, 0.94>;

plane {
    y, -1.9197
    texture {
        pigment {
            checker grid_color_light, grid_color_dark
            scale 1.1697
        }
        finish {
            ambient 0.25
            diffuse 0.55
            specular 0.08
            roughness 0.04
        }
    }
}

// Softer finish settings
#declare charge_finish = finish {
    ambient 0.25
    diffuse 0.55
    specular 0.35
    roughness 0.02
    metallic 0.05
    phong 0.35
    phong_size 50
    brilliance 1.05
}

#declare esp_finish = finish {
    ambient 0.28
    diffuse 0.55
    specular 0.18
    roughness 0.05
    brilliance 1.0
}

sphere {
  <0.0000, 0.0000, 0.0000>, 0.0600
  texture {
    pigment { rgbf <0.972, 0.977, 0.980, 0.70> }
    finish { charge_finish }
  }
}

sphere {
  <0.0000, 0.0000, 0.0000>, 0.0600
  texture {
    pigment { rgbf <0.412, 0.513, 0.628, 0.70> }
    finish { charge_finish }
  }
}

sphere {
  <0.0000, 0.0000, 0.0000>, 0.0600
  texture {
    pigment { rgbf <0.983, 0.972, 0.966, 0.70> }
    finish { charge_finish }
  }
}

sphere {
  <0.0000, 0.0000, 1.2521>, 0.0600
  texture {
    pigment { rgbf <0.834, 0.909, 0.948, 0.70> }
    finish { charge_finish }
  }
}

sphere {
  <0.0000, 0.0000, 1.1843>, 0.0600
  texture {
    pigment { rgbf <0.504, 0.675, 0.823, 0.70> }
    finish { charge_finish }
  }
}

sphere {
  <0.0000, 0.0000, 1.1470>, 0.0600
  texture {
    pigment { rgbf <0.934, 0.959, 0.972, 0.70> }
    finish { charge_finish }
  }
}

sphere {
  <0.0000, 0.0000, -1.2521>, 0.0600
  texture {
    pigment { rgbf <0.834, 0.909, 0.948, 0.70> }
    finish { charge_finish }
  }
}

sphere {
  <0.0000, 0.0000, -1.1843>, 0.0600
  texture {
    pigment { rgbf <0.504, 0.675, 0.823, 0.70> }
    finish { charge_finish }
  }
}

sphere {
  <0.0000, 0.0000, -1.1470>, 0.0600
  texture {
    pigment { rgbf <0.934, 0.959, 0.972, 0.70> }
    finish { charge_finish }
  }
}

