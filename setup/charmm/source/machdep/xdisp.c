/**************************************************

Interface from CHARMM and LIGHT graphics to Xlib

          by Milan Hodoscek 
            November 1993

   ---------------------------------------------------------------

CHARMM interface modified extensively by Rick Venable
                         rvenable@deimos.cber.nih.gov

Nov 1994 change summary:

  - added tests for displays with immutable colormaps (StaticColor
    visual); fallback to sharing the closest RGB color in the read-only
    colormap, avoiding wacky colors when there's no private colormap

  - for PseudoColor visuals, corrected colormap handling; a private
    colormap is created, and assigned to the client's window;
    (no XInstallColormap() calls required, mwm or vuewm does this)

  - drawing done to a pixmap, then copied to the window for a software
    double buffering effect; Xflush() called only upon drawing completion,
    increasing communications efficiency to the X server; xshowup routine
    used for XCopyArea() call, with the window as the destination

  - stereo pair clipping added with XSetClipRectangles() calls

  - symbol fonts added, same 4 sizes as the labeling and text fonts

**************************************************/
// BIOVIA Code Start : Fix for Windows
#ifndef WIN32
// BIOVIA Code End
#ifdef xdisplay

#include <stdio.h>
#include <stdlib.h>

#ifndef intel

#ifdef CHARMM_GNUC
#undef hpux  /* This is for running GNU compilers on hpux */
#endif
 

#include <X11/Xlib.h>
#include <X11/Xutil.h>

#define COLOR_OFFSET 60
#define NAME " XCHARMM v 0.2 "

int *pixels ;         /*storage for image data */


Display *display;
int screen;
Window win, rootw;
Pixmap pxm;
Visual *visual ;
int depth, class;
Colormap cmap;
XColor colorcell_defs[256];
GC gc;
XGCValues xgcv;

XFontStruct *font1, *font2, *font3, *font4, *curr_font ;
XFontStruct *symf1, *symf2, *symf3, *symf4 ;
char *fontn1 = "-adobe-times-bold-r-normal--12-120-75-75-p-67-iso8859-1" ;
char *fontn2 = "-adobe-times-bold-r-normal--14-140-75-75-p-77-iso8859-1" ;
char *fontn3 = "-adobe-times-bold-r-normal--18-180-75-75-p-99-iso8859-1" ;
char *fontn4 = "-adobe-times-bold-r-normal--24-240-75-75-p-132-iso8859-1" ;
char *symfn1 = "-adobe-symbol-medium-r-normal--12-120-75-75-p-74-adobe-fontspecific" ;
char *symfn2 = "-adobe-symbol-medium-r-normal--14-140-75-75-p-85-adobe-fontspecific" ;
char *symfn3 = "-adobe-symbol-medium-r-normal--18-180-75-75-p-107-adobe-fontspecific" ;
char *symfn4 = "-adobe-symbol-medium-r-normal--24-240-75-75-p-142-adobe-fontspecific" ;

char *display_name = NULL;
XImage *ima;
int xsize, ysize, pntrwarp;

#if ibmrs || hpux
void xinitdisp(x,y,d,ncol,fcolor)
#else
void xinitdisp_(x,y,d,ncol,fcolor)
#endif
     int *x, *y, *d, *ncol, *fcolor ;
{
  XVisualInfo *vinfo, rvinfo;
  unsigned int vinfomask;
  int numvis, best, ncells;
  XSetWindowAttributes xswa;
  XTextProperty winttl;
  Status stat;
  unsigned int xswamask;
  XWindowAttributes wattr ;
  int i, j, k, ind ;
  void setcolorm() ;

  /* connect to X server */
  if ((display=XOpenDisplay(display_name)) == NULL) {
    printf("error: cannot connect to X server %s\n",
           XDisplayName(display_name));
    exit(1);
  }

/*  if ( *d == 24 ) rvinfo.class = TrueColor ;  */

  rvinfo.screen = DefaultScreen(display);
  rvinfo.class = TrueColor ;
  vinfomask = VisualDepthMask | VisualScreenMask | VisualClassMask ;
  vinfo = XGetVisualInfo(display, vinfomask, &rvinfo, &numvis);
  /* First find bpp number, then follow the old procedure
     to find the visual */
  rvinfo.depth = 32 ;
  vinfo = XGetVisualInfo(display, vinfomask, &rvinfo, &numvis);
  if ( numvis == 0 ) {
    rvinfo.depth = 24 ;
    vinfo = XGetVisualInfo(display, vinfomask, &rvinfo, &numvis);
    if ( numvis == 0 ) {
      rvinfo.depth = 16 ;
      vinfo = XGetVisualInfo(display, vinfomask, &rvinfo, &numvis);
      if ( numvis == 0 ) {
        rvinfo.depth = 15 ;
        vinfo = XGetVisualInfo(display, vinfomask, &rvinfo, &numvis);
      }
    }
  }
  if ( numvis == 0 )
    {
      printf("No TrueColor visual available on %s, trying PseudoColor\n",
             XDisplayName( display_name ) ) ;
      rvinfo.depth = 8 ;
      rvinfo.class = PseudoColor ;
      vinfo = XGetVisualInfo(display, vinfomask, &rvinfo, &numvis);
      if ( numvis == 0 )
        {
          printf("No PseudoColor visual available on %s, trying StaticColor\n",
                 XDisplayName( display_name ) ) ;
          rvinfo.class = StaticColor ;
          vinfo = XGetVisualInfo(display, vinfomask, &rvinfo, &numvis);
          if ( numvis == 0 )
            {
              printf("No StaticColor visual available on %s, giving up\n",
                     XDisplayName( display_name ) ) ;
              XCloseDisplay( display ) ;
              exit(1) ;
            }
          else
            {
              visual = vinfo->visual;  
              screen = vinfo->screen;
              ncells = vinfo->colormap_size;
              depth = vinfo->depth;
              class = vinfo->class;
              rootw = RootWindow(display, screen);
              cmap = XCreateColormap(display, rootw, visual, AllocNone);
            }
        }
      else
        {
          visual = vinfo->visual;  
          screen = vinfo->screen;
          ncells = vinfo->colormap_size;
          depth = vinfo->depth;
          class = vinfo->class;
          rootw = RootWindow(display, screen);
          cmap = XCreateColormap(display, rootw, visual, AllocAll);
        }
    }
    else
      {
        visual = vinfo->visual;  
        screen = vinfo->screen;
        ncells = vinfo->colormap_size;
        depth = vinfo->depth;
        class = vinfo->class;
        rootw = RootWindow(display, screen);
        cmap = XCreateColormap(display, rootw, visual, AllocNone);
      }

  XFree(vinfo);
  XFlush(display);
  xswa.background_pixel = 0;
  xswa.border_pixel     = 1;
  xswa.colormap         = cmap;
  xswamask = CWBackPixel | CWBorderPixel | CWColormap;

  xsize  = *x ; ysize = *y;
  win = XCreateWindow(display,rootw,0,0,xsize,ysize,1,
                      depth,InputOutput,visual,xswamask,&xswa);
  gc = XCreateGC(display, win, 0, &xgcv);
  setcolorm(ncol,fcolor) ;

/*  XSetForeground(display,gc,200); */

/*  stat = XStringListToTextProperty( " XCHARMM v 0.2 ",1,winttl);
  XSetWMName( display, win, winttl ) ; */

  XStoreName( display, win, NAME );
  XMapRaised( display, win);
  
/* move window to the top left corner */
  XMoveWindow(display,win,10,20);

/* setup the off-screen pixmap drawable */
  pxm = XCreatePixmap(display, win, xsize,ysize, depth);

/* wait for slow remote displays to have window displayed */
  XSync(display, False);
}

#if ibmrs || hpux  
void xshowup()
#else
void xshowup_()
#endif
{
  XRaiseWindow( display, win) ;
  XClearWindow( display, win) ;
/*  XSetInputFocus( display, win, RevertToPointerRoot, CurrentTime ); */
/*  XWarpPointer( display, None, win, 0,0, 0,0, xsize-1,ysize-1 ); */
  if ( pntrwarp == 1 ) {
    XWarpPointer( display, None, win, 0,0, 0,0, xsize-1,ysize-1 ); 
  }
  XCopyArea( display, pxm, win, gc, 0,0, xsize,ysize, 0,0) ;
  XFlush(display);
}


#if ibmrs || hpux
void xfontinit()
#else
void xfontinit_()
#endif
{
/* Open 8 fonts */

  if ( ( font1 = XLoadQueryFont(display,fontn1) ) == NULL ){
    printf("Cannot open font: %s\n", fontn1) ;
  }
  if ( ( font2 = XLoadQueryFont(display,fontn2) ) == NULL ){
    printf("Cannot open font: %s\n", fontn2) ;
  }
  if ( ( font3 = XLoadQueryFont(display,fontn3) ) == NULL ){
    printf("Cannot open font: %s\n", fontn3) ;
  }
  if ( ( font4 = XLoadQueryFont(display,fontn4) ) == NULL ){
    printf("Cannot open font: %s\n", fontn4) ;
  }
  if ( ( symf1 = XLoadQueryFont(display,symfn1) ) == NULL ){
    printf("Cannot open font: %s\n", symfn1) ;
  }
  if ( ( symf2 = XLoadQueryFont(display,symfn2) ) == NULL ){
    printf("Cannot open font: %s\n", symfn2) ;
  }
  if ( ( symf3 = XLoadQueryFont(display,symfn3) ) == NULL ){
    printf("Cannot open font: %s\n", symfn3) ;
  }
  if ( ( symf4 = XLoadQueryFont(display,symfn4) ) == NULL ){
    printf("Cannot open font: %s\n", symfn4) ;
  }

}

#if ibmrs || hpux
void xselfont(i)
#else
void xselfont_(i)
#endif
     int *i ;
{
  switch( *i ) {
  case 1 : { XSetFont(display,gc,font1->fid) ; break ; }
  case 2 : { XSetFont(display,gc,font2->fid) ; break ; }
  case 3 : { XSetFont(display,gc,font3->fid) ; break ; }
  case 4 : { XSetFont(display,gc,font4->fid) ; break ; }
/* symbol fonts of matching point sizes */
  case 11 : { XSetFont(display,gc,symf1->fid) ; break ; }
  case 12 : { XSetFont(display,gc,symf2->fid) ; break ; }
  case 13 : { XSetFont(display,gc,symf3->fid) ; break ; }
  case 14 : { XSetFont(display,gc,symf4->fid) ; break ; }
  default : { printf("Wrong pointer to the font.\n") ; }
  }
}

#if ibmrs || hpux  
void xtext(text,len,x,y)
#else
void xtext_(text,len,x,y)
#endif
     char *text ;
     int *len, *x, *y ;
{
  XDrawString(display,pxm,gc,*x,*y,text,*len) ;
}

#if ibmrs || hpux
void xclear()
#else
void xclear_()
#endif
{
  XSetForeground(display,gc,colorcell_defs[COLOR_OFFSET].pixel) ;
  XFillRectangle(display,pxm,gc, 0,0,xsize,ysize) ;
}

#if ibmrs || hpux
void xdispoff()
#else
void xdispoff_()
#endif
{
  XCloseDisplay(display) ;
}

#if ibmrs || hpux
void xsetwarp(pw)
#else
void xsetwarp_(pw)
#endif
int *pw;
{
  pntrwarp = *pw;
}

#if ibmrs || hpux
void xlinewidth(width)
#else
void xlinewidth_(width)
#endif
     int *width;
{
  xgcv.line_width = *width ;
  XChangeGC(display,gc,GCLineWidth,&xgcv) ;
}

#if ibmrs || hpux
void xlinestyle(flag)
#else
void xlinestyle_(flag)
#endif
     int *flag ;
{
  xgcv.line_style = LineSolid ;
  if ( *flag != 0 ) xgcv.line_style = LineOnOffDash ;
  XChangeGC(display,gc,GCLineStyle,&xgcv) ;

}

#if ibmrs || hpux
void xcolor(col)
#else
void xcolor_(col)
#endif
     int *col ;
{
  XSetForeground(display,gc,colorcell_defs[*col + COLOR_OFFSET].pixel) ;
}

#if ibmrs || hpux  
void xclipdrw(iflg)
#else
void xclipdrw_(iflg)
#endif
    int *iflg;
{
    int ixcen ;
    XRectangle cliprect[1];

    ixcen = xsize / 2 ;
    if ( *iflg < 0 ) {
      cliprect[0].x = 0 ; cliprect[0].y = 0 ;
      cliprect[0].width = ixcen ; cliprect[0].height = ysize ;
      }
    else if ( *iflg > 0 ) {
      cliprect[0].x = ixcen ; cliprect[0].y = 0 ;
      cliprect[0].width = ixcen ; cliprect[0].height = ysize ;
      }
    else
      {
      cliprect[0].x = 0 ; cliprect[0].y = 0 ;
      cliprect[0].width = xsize ; cliprect[0].height = ysize ;
      }

    XSetClipRectangles( display, gc, 0,0, cliprect, 1, Unsorted ) ;
}

#if CHARMM_GNUC /* No support for integer*2 yet! */
void toint2_(in,out,count)
     int *in, *count;
     short *out ;
{
  int i;
  for ( i = 0 ; i < (*count) ; i++ ) out[i] = (short) in[i] ;
}
#endif

#if ibmrs || hpux  
void xmultiline(xy,count)
#else
void xmultiline_(xy,count)
#endif
     XSegment *xy;
     int *count;
{
  XDrawSegments(display,pxm,gc,xy,*count) ;  
}

#if ibmrs || hpux
void xfarc(x,y,r)
#else
void xfarc_(x,y,r)
#endif
     int *x, *y, *r ;
{
  int xx, yy, w ;
  xx = *x - *r ; yy = *y - *r ;
  w = 2 * *r ; 
  XFillArc(display,pxm,gc,xx,yy,w,w,0,360*64);
}

void setcolorm(ncol,colmap)
     int *ncol, *colmap ;
{
  int ind, i, j, k ;
  unsigned long red, green, blue ;
  Colormap dcmap;

/* copy the default colors */
  if (class == PseudoColor) {
    dcmap = XDefaultColormap( display, screen );
    XQueryColors( display, dcmap, colorcell_defs, 256 );
    XStoreColors( display, cmap, colorcell_defs, 256);
    }
/* set values for the true colormap to be used for the image */
  ind = COLOR_OFFSET;
  for (i=0; i < *ncol; i++){
    red =  256 * ( ( colmap[i] >> 16 ) & 0xff ) ;
    green = 256 * ( ( colmap[i] >> 8 ) & 0xff ) ;
    blue = 256 * ( colmap[i] & 0xff ) ;
    colorcell_defs[ind].pixel = ind ;
    colorcell_defs[ind].blue = blue ;
    colorcell_defs[ind].green = green ;
    colorcell_defs[ind].red = red ;
    colorcell_defs[ind].flags = DoRed | DoGreen | DoBlue ;
/*
    printf("Colormap[%d]=%d : red = %d green = %d blue = %d\n", ind, colmap[i],
           colorcell_defs[ind].red, colorcell_defs[ind].green, colorcell_defs[ind].blue);
*/
    ind++;
  }
  
/* store the colors in the colormap for PseudoColor visual */
  if (class == PseudoColor) {
    XStoreColors(display, cmap, colorcell_defs, 256);
    XSetWindowColormap(display, win, cmap);
  }

/* get the closest match for {Static,True}Color visual */
  else {
    ind = COLOR_OFFSET;
    for (i=0; i < *ncol; i++) {
      k = XAllocColor( display, cmap, &colorcell_defs[ind]) ;
      ind++ ;
    }
  }
}

/*    --- end of routines used by CHARMM graphx ---
        --- remaining used for LIGHT images ---       */

void putimage(pix)
int *pix;
{
  char *pixls ;
  int i, ipt, red, green, blue, red6, green6, blue6 ;
  void setcolors(), convert24to8(), convert24to8a() ;

  pixels = pix ;

  if ( depth == 8 ) {
    pixls = (char *) malloc(xsize*ysize) ;
    convert24to8a(pixels,pixls) ;
    ima = XCreateImage(display, visual, depth, ZPixmap, 0,
                       pixls, xsize, ysize, 8, 0);
  } else
    ima = XCreateImage(display, visual, depth, ZPixmap, 0,
		       (char *) pixels, xsize, ysize, 32, 0);

  XPutImage(display, win, gc, ima, 0, 0, 0, 0, xsize, ysize);
  XFlush(display);
}

void xevent()
{
  char c;
  int i;
  printf("Put X event stuff here\nFor now just hit return!");
  i=scanf("%c",&c);
  XCloseDisplay(display) ;

}
void setcolorsa()
{
  int ind, i, j, k ;

  /* set values for the true colormap to be used for the image */
  ind = COLOR_OFFSET;
  for(i = 0; i < 6; i++){
    for(j = 0; j < 6; j++){
      for(k =0; k < 6; k++){
        colorcell_defs[ind].pixel = ind ;
        colorcell_defs[ind].blue = 13106 * k ;
        colorcell_defs[ind].green = 13106 * j ;
        colorcell_defs[ind].red = 13106 * i ;
        colorcell_defs[ind].flags = DoRed | DoGreen | DoBlue ;
        ind++;
      }
    }
  }
  
/* store the colors in the colormap */

  if (depth == 8){
    XSetWindowColormap(display, win, cmap);
    XStoreColors(display, cmap, colorcell_defs, 256);
/*    XInstallColormap(display, cmap); */
  }
  
}

/*
** This is based on Paul Heckbert's paper 
** "Color Image Quantization for Frame Buffer Display",
**  SIGGRAPH '82 Proceedings, page 297.
** and implementation in the 'pbmplus' package written by Jef Poskanzer
*/

/* ppmquant.c - quantize the colors in a pixmap down to a specified number
**
** Copyright (C) 1989, 1991 by Jef Poskanzer.
**
** Permission to use, copy, modify, and distribute this software and its
** documentation for any purpose and without fee is hereby granted, provided
** that the above copyright notice appear in all copies and that both that
** copyright notice and this permission notice appear in supporting
** documentation.  This software is provided "as is" without express or
** implied warranty.
*/

#define MAXCOLORS 32767  /* X intensities */
#define HASH_SIZE 6553

void convert24to8(p24, p8)
     int *p24 ;
     char *p8 ;
{
  int colors ;
  unsigned char maxv, nmaxv, r, g, b ;
  int i, xy, hash, *pp ;
  xy = xsize * ysize ;

  pp = p24 ;

  for ( i = 0 ; i < xy ; i++ ){
    r = (p24[i] >> 16) & 0xff ; g = (p24[i] >> 8) & 0xff ; b = p24[i] & 0xff ;
    hash = (((int) r * 33023 + (int) g * 30013 + (int) b * 27011 )
            & 0x7fffffff ) % HASH_SIZE ;
  }
}


void convert24to8a(p24, p8)
     int *p24 ;
     char *p8 ;
{
  int ipt, i, red, green, blue, red6, green6, blue6;

  setcolorsa() ;

  ipt = 0 ; 
  for(i=0;i<xsize * ysize;i++){
    red = (p24[i] >> 16) & 0xff ; 
    green = (p24[i] >> 8) & 0xff ;
    blue = p24[i] & 0xff ;
    red6 =  (red * 5 + 127) / 256 ;
    green6 = (green * 5 + 127) / 256 ;
    blue6 =  (blue * 5 + 127) / 256 ;
    p8[ipt++] = red6 * 36 + green6 * 6 + blue6 + COLOR_OFFSET ;
  } 
}

#endif
#endif
// BIOVIA Code Start : Fix for Windows
#endif
// BIOVIA Code End
