/*CHARMM Element source/machdep/socket.c

 BSD unix socket routines for parallel CHARMM

              by Milan Hodoscek

                December 1992

**********************************************/
// BIOVIA Code Start : Fix for Windows
#ifndef WIN32
// BIOVIA Code End

#include <stdio.h>

#ifndef intel
# ifndef t3d
#  ifndef cray
#   ifndef terra

#    include <unistd.h>
#    include <stdlib.h>
#    include <string.h>
#    include <sys/time.h>
#    include <sys/resource.h>

// socket include  stuff
#    include <sys/types.h>
#    include <sys/socket.h>
#    include <netinet/in.h>
#    include <netinet/tcp.h>
#    include <netdb.h>
#    include <signal.h>
#    include <sys/time.h>
// end socket include stuff

#    if CHARMM_GNUC
#     include <errno.h>
#     undef hpux
#    endif

extern int errno ;

// global data

#define SIMULATE 1 // = 1 if running simulator...
#define MAXNODE 16
#define HOSTLEN 50
#define SOCK_BIGBUF  32768  // This is maximum?!? I/O socket??
// #define SOCK_BIGBUF  20000 // This works better with boomerang.c

int me, numnod;
char nodes[MAXNODE][10], myhost[HOSTLEN], hosts[MAXNODE][HOSTLEN];
char tmphost[MAXNODE][HOSTLEN], tmpaddr[MAXNODE][HOSTLEN] ;

int sstr[MAXNODE], ls ; // socket stream file descriptors
struct hostent *hinfo[MAXNODE] ;
struct sockaddr_in addr[MAXNODE] ;
int port[MAXNODE] ;

FILE *fdeb ; char ndeb[50];

#if hpux || ibmrs
void getppr(int *p)
#else
void getppr_(int *p)
#endif
//int *p;
{
  *p = getpriority(PRIO_PROCESS,0);
}

#if hpux || ibmrs
void setppr(int *p)
#else
void setppr_(int *p)
#endif
//int *p;
{
  setpriority(PRIO_PROCESS,0,*p);
}

#if hpux || ibmrs
double dclock()
#else
double dclock_()
#endif
{
  // Routine which reports elapsed time in micro second precision

  struct timeval t; struct timezone tz;
  gettimeofday(&t, &tz) ;
  // printf("sec=%d,usec=%d\n",t.tv_sec,t.tv_usec);
  return ( (double) ( t.tv_sec + t.tv_usec / 1000000.0 ) ) ;
}


// #    if chmsocket

// Commented out by APH 10/20/2014
/* #if hpux || ibmrs
void initsock(m, n, fport)
#else
void initsock_(m, n, fport)
#endif
int *m, *n, *fport;
{
  int i, alen, alen1 ;

  void nodelay(), socbuf() ;

  me = *m;
  numnod = *n;

  for(i=0;i<MAXNODE;i++) sprintf(nodes[i],"NODE%d",i);
  for(i=0;i<numnod;i++) strcpy(hosts[i], getenv(nodes[i]));
  strcpy(myhost,hosts[me]);

  alen = alen1 = sizeof(struct sockaddr_in) ;
  // clear addr structures
  for(i=0;i<numnod;i++)
    memset((char *)&addr[i], 0, alen);

  // all nodes create one listen socket to get node's port number
  addr[me].sin_family = AF_INET ;
  addr[me].sin_addr.s_addr = INADDR_ANY ;
  ls = socket(AF_INET, SOCK_STREAM, 0) ; nodelay(ls) ;
  if ( ls == -1 ){
    perror("listen socket"); exit(111) ;
  }
  socbuf(ls) ;

  addr[me].sin_port = 0 ;

  if(bind(ls,&addr[me],alen) == -1){
    perror("listen socket binding"); exit(112);
  }

  if(getsockname(ls,(struct sockaddr_in *)&addr[me],&alen1) == -1){
    perror("after listen socket getsockname"); exit(113);
  }

  if(listen(ls,1) == -1){
    perror("listen for listen socket"); exit(114);
  }

  *fport = port[me] = addr[me].sin_port ;

//  printf("node = %d, port = %d\n", me, addr[me].sin_port);

  sprintf(ndeb,"debug.%d",me);
// fdeb = fopen(ndeb,"w"); setbuf(fdeb, (char *)NULL) ;
  fdeb = fopen("/dev/null","w"); setbuf(fdeb, (char *)NULL) ;
  fprintf(fdeb,"=============================\n");
  fprintf(fdeb,"This is debug file from node%d\n",me);
  fprintf(fdeb,"=============================\n");
  fprintf(fdeb,"numnode=%d\n---------\n",numnod);
  fprintf(fdeb,"node = %d, port = %d, addr = %d\n",
	  me, addr[me].sin_port, addr[me].sin_addr.s_addr);
}

// Commented out by APH 10/20/2014
#if hpux || ibmrs
void fconnect()
#else
void fconnect_()
#endif
{
  // This routine connects nodes among themselves
  // We never need sstr[me], because we are not sending
  // data to self: it is listen socket ls, which is
  // reported to other nodes for accepting connections

  int m, k, alen, alen1, l, temp_socket, repeat, flag ;
  void recvall(), nodelay(), socbuf() ;
  int nodinf() ;

  alen = alen1 = sizeof(struct sockaddr_in) ;
  l = 0 ;

//  printf("me=%d, in fconnect\n", me);
  fprintf(fdeb,"me=%d, in fconnect\n", me);


  // first CONNECT to nodes 0, ..., me-1
  for(k=0;k<me;k++){

    sstr[k] = socket(AF_INET, SOCK_STREAM, 0) ;
    nodelay(sstr[k]); socbuf(sstr[k]);

    if(k == 0){
      addr[k].sin_port = port[k] = atoi(getenv("PORT"));
    } else {
      // Receive port numbers [1],...,[me-1] from NODE0
      recvall(sstr[0],&port[k],sizeof(port[k])) ; addr[k].sin_port = port[k];
      fprintf(fdeb,"Receiving port[k] successful:port[%d]=%d\n",k,port[k]);
    }
    hinfo[k] = gethostbyname(hosts[k]);

    addr[k].sin_addr.s_addr =
      ((struct in_addr *)(hinfo[k]->h_addr))->s_addr ;

    addr[k].sin_family = AF_INET ;


//    printf("before connect to NODE%d:me=%d, error=%d, port0=%d,sock0=%d\n",
//	   k,me,errno,port[k],sstr[k]);

    fprintf(fdeb,"before connect to NODE%d:me=%d, error=%d, port0=%d,sock0=%d\n",
	    k,me,errno,port[k],sstr[k]);

    if(connect(sstr[k],&addr[k],alen) == -1){
      fprintf(fdeb,"error in connect to NODE%d:me=%d, error=%d, port0=%d,sock0=%d\n",
	      k,me,errno,port[k],sstr[k]);
      perror("connect problems"); exit(118);
    }

    fprintf(fdeb,"after connect to NODE%d:me=%d, error=%d, port0=%d,sock0=%d\n",
	    k,me,errno,port[k],sstr[k]);

    if (k == 0){
      // Now send port number of my listen socket:
      // allways to NODE0 if me == numnod-1 we also send, although not needed
      l = send(sstr[0], &port[me], sizeof(port[me]), 0) ;
      if ( l != 4 ){
	printf("Sending port[me] unsuccessful:req=%d,actual=%d\n",
	       4,l); exit(144);
      }
      fprintf(fdeb,"Sending port[me] successful\n");
    }
  }


  // ACCEPT connections from nodes me+1,...,numnod-1

  m = 0 ;
  for(k=me+1;k<numnod;k++){
    fprintf(fdeb,"Before accept: k=%d\n",k);
    temp_socket = accept(ls, &addr[k], &alen1) ;
    fprintf(fdeb,"After accept: sstr[%d]=%d\n",k,temp_socket);
    if (temp_socket == -1 ) {
      perror("accept problems");exit(115) ;
    }
    nodelay(temp_socket); socbuf(temp_socket); // set some socket options

    // find out who was it:
    // All nodes execute connect at approximate
    // the same time so this is code is here to match definitions of node
    // names on all nodes:

    hinfo[k] = gethostbyaddr((char *)&addr[k].sin_addr,
			     sizeof(struct in_addr),
			     addr[k].sin_family) ;
    if ( hinfo[k] == NULL ) {
      perror("problems with info"); exit(120);
    }

    if ( SIMULATE != 1 ){
      m=nodinf(hinfo[k]->h_name,addr[k].sin_addr);
      if(m == -1){
	printf("Problems with host names:%d=%s\n",k,hinfo[k]->h_name);
	exit(160);
      }
    } else m = k ;

    if(m == -1){
      printf("Problems with host names:%d=%s\n",k,hinfo[k]->h_name);
      exit(160);
    }
    sstr[m]=temp_socket ;
    fprintf(fdeb,"This is socket: sstr[%d]=%d\n",m,temp_socket);

  }

  // Now each node has all ports and sockets
  // Exception is Node 0 since it has to distribute
  // port numbers to other nodes
  // First receive them:

  if (me == 0){
    for(k=1;k<numnod;k++){
      // get the listen socket port number -- this information
      // is also available from addr[k] -- I am not sure!! -- check
      // it reports different number !? check why!
      recvall(sstr[k],&port[k],sizeof(port[k])) ;
    }

    // send port numbers to the nodes which need them
    for(k=2;k<numnod;k++)
      for(m=1;m<k;m++){
	l = send(sstr[k], &port[m], sizeof(port[m]), 0) ;
	if ( l != 4 ){
	  printf("Problem sending port[%d] to node%d\n",m,k); exit(145);
	}
	fprintf(fdeb,"Sending port[me] successful\n");
      }
  } // if(me == 0)

// printf("me=%d, @end fconnect\n", me);
// for(m=0;m<numnod;m++)printf(", port[%d]=%d",m, port[m]) ;
// printf("\n");

  fprintf(fdeb,"me=%d, @end fconnect\n", me);
  for(m=0;m<numnod;m++)
    fprintf(fdeb,"; soc[%d]=%d, port[%d]=%d",m,sstr[m],m,port[m]) ;
  fprintf(fdeb,"\n");

}

// Commented out by APH 10/20/2014
int nodinf(host,addr)
char host[];
unsigned long addr;
{

  unsigned long s1, s2, s3, s4 ; char laddr[HOSTLEN];
  int k, l,l2, r, i;
//  printf("name of this  node is: %s\n",host);
  s1 = 0xff ; s2 = 0xff00 ; s3 = 0xff0000 ; s4 = 0xff000000 ;
  sprintf(laddr,"%d.%d.%d.%d\n", (addr & s4)>>24, (addr & s3)>>16, (addr & s2)>>8, addr & s1);

//   printf("address of this  node is: %s\n",laddr);
  fprintf(fdeb,"name of this  node is: %s\n",host);
  fprintf(fdeb,"address of this  node is: %s\n",laddr);
  for (k=0;k<numnod;k++){
    l = strlen(hosts[k]);
// names of different length are different. LNI
    l2 = strlen (host);
// and we don't need the full domain name....
    for (i=0; i<l2; i++) {
      if( host[i] == '.') l2=i;
      }
    r=l2-l;
    if ( r == 0) r = strncmp(hosts[k],host,l);
    if ( r == 0 ) return(k);
  }
  return(-1);
}

// Commented out by APH 10/20/2014
void socbuf(s)
int s ;
{
  int is, bufs;

  bufs = SOCK_BIGBUF ;
  if(setsockopt(s, SOL_SOCKET, SO_RCVBUF, &bufs, sizeof(bufs)) !=0){
    perror("setsockopt-rbufs"); exit(151);
  }
  if(setsockopt(s, SOL_SOCKET, SO_SNDBUF, &bufs, sizeof(bufs)) !=0){
    perror("setsockopt-sbufs"); exit(152);
  }
}

// Commented out by APH 10/20/2014
void nodelay(s)
int s;
{
  int lvl, val = 1 ;
  struct protoent *proto = getprotobyname("tcp") ;
  if (proto == (struct protoent *) NULL){
    perror("setsockopt"); exit(150);
  }
  lvl = proto->p_proto;
  if(setsockopt(s, lvl, TCP_NODELAY, &val, sizeof(int)) != 0){
    perror("setsockopt"); exit(153);
  }
}

// Commented out by APH 10/20/2014
void recvall(soc,buf,len)
int soc, len;
char *buf;
{
  int k, m ;
  if( (k=recv(soc,buf,len, 0)) == -1 ){
    perror("recvall>cannot recv");  exit(117);
  }
  while ( k < len ){
    m = recv(soc, &buf[k], len-k, 0); k += m ;
    if ( m == -1 ){
      perror("recvall>cannot recv");  exit(119);
    }
  }
}

// Commented out by APH 10/20/2014
#if hpux || ibmrs
void frecv(from,buf,len)
#else
void frecv_(from,buf,len)
#endif
int *from, *len;
char *buf;
{
  void recvall();
//  printf("frecv>me:from,len=%d:%d %d\n", me, *from, *len);
  recvall(sstr[*from],buf,*len);
}

// Commented out by APH 10/20/2014
#if hpux || ibmrs
void fsend(to,buf,len)
#else
void fsend_(to,buf,len)
#endif
int *to, *len;
char *buf;
{
  int l ;
//  printf("fsend>me:to,len=%d:%d %d\n", me, *to, *len);
  l = send(sstr[*to], buf, *len, 0) ;
  if (l != *len){
    printf("troubles with send:requested=%d, actual=%d\n",
	   *len,l); exit(130);
  }
}

// Commented out by APH 10/20/2014
void fsend_too_compl(to,buf,len)
int *to, *len;
char *buf;
{
  int l, lenb, ls ; char *lbuf;
  // printf("fsend>me:to,len=%d:%d %d\n", me, *to, *len);
  lenb=*len; lbuf = buf ;
  while ( lenb > 0 ){
    ls = ( lenb > SOCK_BIGBUF ) ? SOCK_BIGBUF : lenb ;
    l = send(sstr[*to], lbuf, ls, 0) ;
    if (l != ls){
      printf("troubles with send:requested=%d, actual=%d\n",
	     *len,l); exit(130);
    }
    lbuf += ls ; lenb -= ls ;
  }
}

// Commented out by APH 10/20/2014
void chmshut_()
{
  int i, j ;
  for ( j = 0 ; j < numnod ; j++ ) {
    if ( j != me ){
      i = shutdown(sstr[j], 2) ;
      if ( i == -1 ) {
	perror("chmshut>Troubles with shutdown") ;
	exit(222) ;
      }
      close(sstr[j]) ;
    }
  }
}

// Commented out by APH 10/20/2014
#if hpux || ibmrs
void ffork(pid)
#else
void ffork_(pid)
#endif
int *pid ;
{
//  printf("setpgrp=%d\n", setpgrp());
// On HP-UX vfork() sometimes doesn't work ???
// put hpuxx to avoid calling it
#if hpuxx || Alpha
  *pid=vfork();
#else
  *pid=fork();
#endif
}

// Commented out by APH 10/20/2014
#if hpux || ibmrs
void ffexec(bin, arg0, arg1, arg, arg2)
#else
void ffexec_(bin, arg0, arg1, arg, arg2)
#endif
char *bin, *arg0, *arg1, *arg, *arg2 ;
{
// These are many parameters:
// Problem is that C doesn't know (by some easy way:no pointers?)
// how to get **argv from character*50 argv(50)

  char *a[8];

// Assign pointers:

  a[0] = arg0;           // remsh command name
  a[1] = &arg1[0];       // remote node name
  a[2] = (char *) "-n" ;          // option
  a[3] = (char *) " " ;           // could be -l
  a[4] = (char *) " ";            // could be getenv($USER)
  a[5] = &arg2[0];       // additional environment variables
  a[6] = &arg[0];        // environment variables + executable
  a[7] = (char *) NULL ;

// NOTE:  SGI has problems with a[2-4]
//          - this is hopefully fixed with the correct casting

  (void) execv(bin,a);

  printf("execv ???\n");
// If it comes here something is wrong!
  perror("fexec>after execv") ;

}

// Commented out by APH 10/20/2014
#if hpux || ibmrs
void fgtcwd(p,l,lp)
#else
void fgtcwd_(p,l,lp)
#endif
char *p;
int *l, *lp;
{

  *lp = 0 ;
  if ( getcwd(p,*l) == NULL ) perror("getcwd") ;
  //  if(errno != 0){ perror("getcwd");}
  *lp=strlen(p);
}

#endif // socket

// Commented out by APH 10/20/2014
#if hpux || ibmrs
void nocore()
#else
void nocore_()
#endif
{
  void endc();

  // These signals create core file, so catch them if you don't
  // want it.
  // (From HP-UX Reference Volume 3, section 5, signal(5), p.683)

  signal(SIGQUIT, endc);
  signal(SIGILL , endc);
  signal(SIGTRAP, endc);
  signal(SIGABRT, endc);
  signal(SIGIOT , endc);
#ifdef hpux
  signal(SIGEMT , endc);
  signal(SIGSYS , endc);
#endif
  signal(SIGFPE , endc);
  signal(SIGBUS , endc);
  signal(SIGSEGV, endc);
}

void endc(i,j,k)
int i,j,k ;
{
  printf("This comes from nocore:%d %d %d\n",i,j,k);

  switch ( i )
    {
    case SIGQUIT : printf("%d:SIGQUIT ",me); break; // 3
    case SIGILL  : printf("%d:SIGILL ",me);  break; // 4
    case SIGTRAP : printf("%d:SIGTRAP ",me); break; // 5
    case SIGABRT : printf("%d:SIGABRT or SIGIOT ",me); break; // 6 = SIGIOT
#ifdef hpux
    case SIGEMT  : printf("%d:SIGEMT ",me);  break; // 7
    case SIGSYS  : printf("%d:SIGSYS ",me);  break; // 12
#endif
    case SIGFPE  : printf("%d:SIGFPE ",me);  break; // 8
    case SIGBUS  : printf("%d:SIGBUS ",me);  break; // 10
    case SIGSEGV : printf("%d:SIGSEGV ",me); break; // 11
    default : printf("%d:unknown signal ",me); break;
    }
  perror("from nocore");
  exit(i);
} */

#   endif   /* terra */
#  endif   /* cray */
# endif   /* t3d */
#endif  /* intel */

// BIOVIA Code Start : Fix for Windows
#endif /* WIN32 */
// BIOVIA Code End
