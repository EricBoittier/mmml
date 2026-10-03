#include <string>
#include <cstring>
#include <iostream>
#include <algorithm>

// BIOVIA Code Start : Fix for Windows
#ifdef WIN32
#include <winsock.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")
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

#ifndef HOST_NAME_MAX
#define HOST_NAME_MAX 255 // POSIX MINIMUM
#endif

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
    char buf[HOST_NAME_MAX + 1];
    if (gethostname(buf, sizeof(buf)) != 0) {
        return "unknown";
    }
    buf[HOST_NAME_MAX] = '\0';

#if defined(_WIN32)
    // GetComputerNameExA gives FQDN directly on Windows
    char fqdn[256];
    DWORD size = sizeof(fqdn);
    if (GetComputerNameExA(ComputerNameDnsFullyQualified, fqdn, &size)) {
        return std::string(fqdn);
    }
#endif

    // gethostbyname returns canonical name in h_name
    struct hostent *he = gethostbyname(buf);
    if (he && he->h_name && he->h_name[0] != '\0') {
        return std::string(he->h_name);
    }

    return std::string(buf);
}
#endif /* STATIC != 1 */

// os details will be stored in sy which is overwritten
// sy has max allocated size *lsy upon entry
// fully qualified domain name will be stored in hn also overwritten
// hn has max allocated size *lhn upon entry
extern "C" void uninf(char * sy, int lsy, char * hn, int lhn)
{
  std::string errArgs =
    "uninf> null argument passed  to uninf function"; 

  if (lsy <= 1 || sy == NULL) {
    reportError(errArgs);
    return;
  }

  std::string osname = getOSName();
  if (osname == "") {
    sy[0] = '\0';
  } else {
    size_t nsy = std::min(osname.length(), (size_t) (lsy - 1));
    memcpy(sy, osname.c_str(), nsy);
    sy[nsy] = '\0';
  }

  // in case of hostname failure and early return
  if (lhn <= 1 || hn == NULL) {
    reportError(errArgs);
    return;
  }

// BIOVIA Code Start : Fix for Windows
#if STATIC == 1 || defined(WIN32)
// BIOVIA Code End
  std::string hname = "";
#else
  std::string hname = getFQDN();
#endif

  if (hname == "") {
    hn[0] = '\0';
  } else {
    size_t nhn = std::min(hname.length(), (size_t) (lhn - 1));
    memcpy(hn, hname.c_str(), nhn);
    hn[nhn] = '\0';
    hname = "@" + hname;
  }

  osname += hname;
  if (osname == "" || osname == "@") {
    sy[0] = '\0';
  } else {
    size_t nsy = std::min(osname.length(), (size_t) (lsy - 1));
    memcpy(sy, osname.c_str(), nsy);
    sy[nsy] = '\0';
  }
}
