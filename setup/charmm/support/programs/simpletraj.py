#!/usr/bin/env python

from optparse import OptionParser
from array import array
import struct, string

class CHARMMTrjFile(file):
    """
    This class reads and writes Fortran unformatted
    data files. It is based off of code in the SciPy 
    cook book by Neil Martinsen-Burrell, but heavily modified 
    to make it more convenient to deal with CHARMM stuff.

    The original copyright notice of the SciPy cook book code is:

    Copyright 2008, 2009 Neil Martinsen-Burrell

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in
    all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
    THE SOFTWARE.

    Big fat note: because of the way we keep track of
    frames, you can only read the dcd file from the
    beginning.
    """

    def _read_record(self):
        if self.eightb:
            rlen = 8
        else:
            rlen = 4
        sz = struct.unpack('i', self.read(rlen))[0]
        dt = self.read(sz)
        ck = struct.unpack('i', self.read(rlen))[0]
        if ck != sz:
            raise IOError("Record size and checksum do not match")

        return dt

    def _write_record_nomark(self, rec):
        floatarr = array('f', rec)
        floatarr.tofile(self)

    def _write_record(self, rec):
        sz = len(rec)*4 # data written as 4 byte floats (SP)
        self.write(struct.pack('i',sz))
        floatarr = array('f', rec)
        floatarr.tofile(self)
        self.write(struct.pack('i',sz))

    def _read_frame(self):
        if self.ftype != 'simple':
            if not self.hdr_read:
                raise IOError('Header not read')
            if self.has_xtl:
                self._read_record() # knock off the crystal stuff

        xdat = self._read_record()
        ydat = self._read_record()
        zdat = self._read_record()
     
        x = array('f')
        y = array('f')
        z = array('f')
        x.fromstring(xdat)
        y.fromstring(ydat)
        z.fromstring(zdat)

        # ToDo: put a check here to make sure that the number of
        # values returned matches with self.ndegf

        return x, y, z

    def _write_frame(self, x, y, z):
        if self.ftype != 'simple':
            raise NotImplementedError('Cannot write anything but simple trajectory files yet')
        self._write_record_nomark(x)
        self._write_record_nomark(y)
        self._write_record_nomark(z)

    def read_header(self):
        if self.ftype == 'simple':
            raise NotImplementedError('Simple files do not have headers')
        hdrdat = self._read_record()
        self.dcdtype = string.join(struct.unpack('cccc', hdrdat[:4]),'')
        if self.verbose:
            print 'dcd type is %s' % self.dcdtype
        if self.eightb:
            icntrl = struct.unpack('qqqqqqqqqqqqqqqqqqqq', hdrdat[4:85])
        else:
            icntrl = struct.unpack('iiiiiiiiiiiiiiiiiiii', hdrdat[4:85])
        self.nFrames = icntrl[0]
        self.nPriv   = icntrl[1]
        self.nsavc   = icntrl[2]
        self.nSteps  = icntrl[3]
        self.ndegf   = icntrl[7]
        self.has_xtl = icntrl[10] == 1
        self.has_4d  = icntrl[11] == 1
        if self.has_4d:
            raise NotImplementedError('4-dimensional trajectories are not supported')

        # read the record containing the title
        titlerc = self._read_record()
        titleln = array('c')
        titleln.fromstring(titlerc)
        titleln = [c for c in titleln if c in string.printable]
        self.title = string.join(titleln,'')

        # read the record containing Natom
        natmdat = self._read_record()
        self.natom = struct.unpack('i', natmdat)
        
        if self.verbose:
            print '--- trajectory file information ---'
            print 'Number of atoms  = %d' % self.natom
            print 'Number of frames = %d' % self.nFrames
            print 'Save frquency    = %d' % self.nsavc
            print 'Number of steps  = %d' % self.nSteps
            print 'Number of DOF    = %d' % self.ndegf
            print 'XTL Data?        = %s' % self.has_xtl
            for line in self.title.split('*'):
                if line: print "TITLE: %s" % line.strip()
            print '-----------------------------------'
         
        self.hdr_read = True

    # make this into an iterator
    def next(self):
        self.cur_frame += 1 # the number of the frame we're going to read
        if self.ftype != 'simple' and self.cur_frame > self.nFrames:
            raise StopIteration
        try:
           x, y, z = self._read_frame()
        except EOFError, e:
            
            raise StopIteration
        return x, y, z

    def __init__(self, fname, *args, **kwargs):
        file.__init__(self, fname, *args)

        if kwargs.has_key('eightb'):
            self.eightb = kwargs['eightb']
        else:
            self.eightb = False

        self.ftype = 'simple'
        if kwargs.has_key('mode'):
            self.ftype = kwargs['mode']
        else:
            self.ftype = 'simple'
        if kwargs.has_key('verbose'):
            self.verbose = kwargs['verbose']
        else:
            self.verbose = False
        self.hdr_read = False
        
        self.cur_frame = 0

if __name__ == '__main__':

    parser = OptionParser()
    parser.add_option('-i', '--input', default='traj.dcd', metavar='FILE', \
                      help='File to read frames from')
    parser.add_option('-o', '--output', default='traj.sdcd', metavar='FILE', \
                      help='File to write simplified trajectory')
    parser.add_option('-8', '--eightbyte', action='store_true', \
                      help='Assume eight byte integers on input trajectory')
    (options, args) = parser.parse_args()

    ifp = CHARMMTrjFile(options.input, 'rb', eightb=options.eightbyte, mode='full', verbose=True)
    ofp = CHARMMTrjFile(options.output, 'wb', mode='simple', verbose=True)

    ifp.read_header()
    cnt = 0
    for x, y, z in ifp:
        cnt += 1
        print "frame %d:" % cnt
        for i in range(len(x)):
            print "Atom %d: %8.4f %8.4f %8.4f" % (i,x[i],y[i],z[i])
        print ''
        ofp._write_record_nomark(x)
        ofp._write_record_nomark(y)
        ofp._write_record_nomark(z)

    ifp.close()
    ofp.close()
