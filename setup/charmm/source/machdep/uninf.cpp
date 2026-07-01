#include <string>
#include <cstring>
#include <iostream>
#include <algorithm>

// BIOVIA Code Start : Fix for Windows
#ifdef WIN32
#include <winsock.h>
#include <cstdio>
#include <cstdlib>
#undef min
#else
// BIOVIA Code End
#include <netdb.h>
#include <sys/utsname.h>
#include <sys/socket.h>
#include <unistd.h>
// BIOVIA Code Start : Fix for Windows
#endif
// BIOVIA Code End

void reportError(std::string errMsg) {
  std::cout << errMsg << std::endl;
}

// get OS and machine details formatted as osname-version(architecture)
std::string getOSName()
{
// BIOVIA Code Start : Fix for Windows
#ifdef WIN32
  char buf[1024];
  FILE *pPipe = _popen("ver", "r");
  if (pPipe != NULL) {
     while (!feof(pPipe)) {
        fgets(buf, 1023, pPipe);
        if (strncmp(buf, "Microsoft", 5) == 0) break;
     }
     fclose(pPipe);
  }
  else {
     return "Windows";
  } 
  return buf;
#else
// BIOVIA Code End
  std::string errMsg =
    "uname> unsuccessful at fetching machine details"; 

  struct utsname inf;
  int unameStatus = uname(&inf);

  std::string
    sysname = "", 
    release = "",
    machine = "";

  if (unameStatus == -1) {
    reportError(errMsg);
    sysname = "unknown";
  } else {
    sysname = inf.sysname;

    release = inf.release;
    release = "-" + release;

    machine = inf.machine;
    machine = "(" + machine + ")";
  }

  std::string osname = sysname + release + machine;
  return osname;
// BIOVIA Code Start : Fix for Windows
#endif
// BIOVIA Code End
}

// BIOVIA Code Start : Fix for Windows
#if STATIC != 1 && !defined(WIN32)
// BIOVIA Code End
// get the first fully qualified domain name provided by getaddrinfo
std::string getFQDN() {
  std::string errMsg =
    "getFQDN> hostname problem: check /etc/hosts or /etc/resolv.conf";

  struct addrinfo hints, * info;

  char hostname[1024];
  hostname[1023] = '\0';
  int status = gethostname(hostname, 1023);
  if (status != 0) {
    reportError(errMsg);
    return "";
  }

  std::memset(&hints, 0, sizeof hints);
  hints.ai_family = AF_UNSPEC; // either IPV4 or IPV6
  hints.ai_socktype = SOCK_STREAM;
  hints.ai_flags = AI_CANONNAME;

  status = getaddrinfo(hostname, "http", &hints, &info);
  if (status != 0 || info == NULL) {
    reportError(errMsg);
    return std::string(hostname);
  }

  // simply return the first one in the info linked list
  std::string fqdn = info->ai_canonname;
  freeaddrinfo(info);
  return fqdn;
}
#endif /* STATIC != 1 */

// os details will be stored in sy which is overwritten
// sy has max allocated size *lsy upon entry
// fully qualified domain name will be stored in hn also overwritten
// hn has max allocated size *lhn upon entry
extern "C" void uninf(char * sy, int * lsy, char * hn, int * lhn)
{
  std::string errArgs =
    "uninf> null argument passed  to uninf function"; 

  if (lsy == NULL || *lsy <= 0) {
    reportError(errArgs);
    return;
  } else if (sy == NULL) {
    *lsy = 0;
    reportError(errArgs);
    return;
  }
  std::string osname = getOSName();

  size_t syLimit = *lsy - 1;
  // leave room for any possible string terminator
  // BIOVIA Code Start : Assign type
  *lsy = (int) std::min(osname.length(), syLimit);
  // BIOVIA Code End

  // in case of hostname failure and early return
  // std::strncpy(sy, osname.c_str(), *lsy); 
  memcpy(sy, osname.c_str(), (*lsy) * sizeof(char)); 

  if (lhn == NULL || *lhn <= 0) {
    reportError(errArgs);
    return;
  } else if (hn == NULL) {
    *lhn = 0;
    reportError(errArgs);
    return;
  }
// BIOVIA Code Start : Fix for Windows
#if STATIC == 1 || defined(WIN32)
// BIOVIA Code End
  std::string hname = "";
#else
  std::string hname = getFQDN();
#endif /* STATIC == 1 */
  
  if (hname == "") {
    *lhn = 0;
    hn[0] = 0x0; // duplicate original behavior of uninf
  } else {
    // leave room for any possible string terminator
    // BIOVIA Code Start : Assign type
    *lhn = (int) std::min(hname.length(), (size_t) *lhn - 1);
    // BIOVIA Code End
    // std::strncpy(hn, hname.c_str(), *lhn);
    memcpy(hn, hname.c_str(), (*lhn) * sizeof(char)); 
    hname = "@" + hname;
  }

  osname += hname;
  // BIOVIA Code Start : Assign type
  *lsy = (int) std::min(osname.length(), syLimit);
  // BIOVIA Code End
  std::strncpy(sy, osname.c_str(), *lsy);
}
