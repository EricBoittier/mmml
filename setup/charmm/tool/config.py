# TODO: find a way to make it clear: 1. press configure, 2. press make
#       write documentation for everything !!!
#       add more tabs so that all of --help is covered
#       prompt for threads for make
#       exec might need to make dirs (build/cmake for example)
#       break into separate modules
#       buttons for autofill/find mpi, gcc, intel
#       comment and document everything
#       add explanation of each tab for users
#       mkl/fftw tab with colfft on/off switch
#       gpu options: on/off for ddg, openmm, fftdock
#       add copyright and license comments
#       break up into modules?
#       add configure button to
#           execute cmake
#           put cmake output in message area
#       add 'make clean' button
#       add make button
#       add ninja vs make picker

import sys
import subprocess

try:
    import tkinter as tk
    from tkinter import ttk
    from tkinter import filedialog
except ImportError:
    print('please install the python module tkinter and try again',
          file=sys.stderr)


class CMakeLine:
    """build and run a cmake command line piece by piece

    Attributes
    ----------
    env_vars : dict
        environment variables to set before running cmake
    cmake_vars : dict
        cmake variables to set on cmake command line with -D
    opts : dict
        command line options for the cmake command
    source_dir : str
        source code path for cmake project, appended to command line
    command_line : tk.StringVar
        continuously updated tk string for widgets to display or copy

    Methods
    -------
    set_source_dir(source_dir)
        set the source_dir attribute and update command_line
    add_env_var(name, value)
        add an environment var to env_vars and update the command line
    rem_env_var(name)
        remove an environment var from env_vars, update command_line
    add_cmake_var(name, value='ON')
        add a cmake var to cmake_vars and update the command line
    rem_cmake_var(name)
        remove a cmake var from cmake_vars, update command_line
    add_opt(name, value='')
        add a command line option to opts, update command_line
    rem_opt(name)
        remove a command line option from opts, update command_line
    config()
        run the cmake command in the build dir (cwd or -B in opts)
    make()
        run << make install >> command in build dir (cwd or -B of opts)
    """
    def __init__(self):
        self.env_vars = dict()
        self.cmake_vars = dict()
        self.opts = dict()
        self.source_dir = str()
        self.command_line = tk.StringVar()
        self.command_line.set(str(self))

    def __str__(self):
        outStrs = list()
        for k, v in self.env_vars.items():
            outStrs.append(k + '=' + str(v))

        outStrs.append('cmake')
        for k, v in self.opts.items():
            if v:
                outStrs.append(k + ' ' + v)
            else:
                outStrs.append(k)

        for k, v in self.cmake_vars.items():
            outStrs.append('-D' + k + '=' + str(v))

        if self.source_dir:
            outStrs.append(self.source_dir)

        outStr = ' '.join(outStrs)
        return(outStr)

    def set_source_dir(self, source_dir):
        self.source_dir = source_dir
        self.command_line.set(str(self))

    def add_env_var(self, name, value):
        self.env_vars[name] = value
        self.command_line.set(str(self))

    def rem_env_var(self, name):
        val = None
        if name in self.env_vars:
            val = self.env_vars[name]
            del self.env_vars[name]

        self.command_line.set(str(self))
        return val

    def add_cmake_var(self, name, value='ON'):
        self.cmake_vars[name] = value
        self.command_line.set(str(self))

    def rem_cmake_var(self, name):
        val = None
        if name in self.cmake_vars:
            val = self.cmake_vars[name]
            del self.cmake_vars[name]

        self.command_line.set(str(self))
        return val

    def add_opt(self, name, value=''):
        self.opts[name] = value
        self.command_line.set(str(self))

    def rem_opt(self, name):
        val = None
        if name in self.opts:
            val = self.opts[name]
            del self.opts[name]

        self.command_line.set(str(self))
        return val

    def config(self):
        result = subprocess.run(str(self), shell=True, capture_output=True)
        output = {'stdout': result.stdout, 'stderr': result.stderr}
        return output

    def make(self):
        build_dir = self.opts.get('-B', None)
        result = subprocess.run('make install',
                                cwd=build_dir.strip('"'),
                                shell=True,
                                capture_output=True)
        output = {'stdout': result.stdout, 'stderr': result.stderr}
        return output


class BuildOpts(tk.Frame):
    """a frame with options related to the build rather than features

    Attributes
    ----------
    parent : tk.Frame
        the containing widget
    cmake : CMakeLine
        the cmake command line to modify
    selection : tk.StringVar
        varible that tracks changes in debug/release radio buttons
    rb_debug : tk.Radiobutton
        build a debug release, eg -O0 -g ...
    rb_release : tk.Radiobutton
        build an accelerated release, eg -O2 ...
    frm_source : PathPicker
        the cmake project source code directory
    frm_build : PathPicker
        a build directory for files generated by cmake
    frm_install : PathPicker
        the installation directory for final compiled bins and libs
    frm_in_place : EnableToggle
        build and install in the source dir?

    Methods
    -------
    update_release()
        callback for when debug/acc radio bs are changed, updates cmake
    update_source_dir()
        callback for when source dir changes, updates cmake
    update_build_dir()
        callback for when build dir changes, updates cmake
    update_install_dir()
        callback for when install dir changes, updates cmake
    """
    def __init__(self, parent, cmake, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)
        self.parent = parent
        self.cmake = cmake
        self.selection = tk.StringVar()

        self.rb_debug = tk.Radiobutton(self, text='debug',
                                       value='debug',
                                       var=self.selection,
                                       command=self.update_release)
        self.rb_debug.pack(fill=tk.X)

        self.rb_release = tk.Radiobutton(self, text='accelerated',
                                         value='release',
                                         var=self.selection,
                                         command=self.update_release)
        self.rb_release.pack(fill=tk.X)
        self.rb_release.invoke()

        self.frm_source = PathPicker(self, 'Source Dir',
                                     command=self.update_source_dir,
                                     want_dir=True)
        self.frm_source.pack()

        self.frm_build = PathPicker(self, 'Build Dir',
                                    command=self.update_build_dir,
                                    want_dir=True)
        self.frm_build.pack()

        self.frm_install = PathPicker(self, 'Install Dir',
                                      command=self.update_install_dir,
                                      want_dir=True)
        self.frm_install.pack()

        self.frm_in_place = EnableToggle(self,
                                         'Build and Install in Source Dir',
                                         self.cmake,
                                         disable_widgets=[self.frm_build,
                                                          self.frm_install],
                                         opt_name='in_place_install')
        self.frm_in_place.pack()

    def update_release(self):
        btype = 'Release'
        if self.selection.get() == 'debug':
            btype = 'Debug'

        self.cmake.add_cmake_var('CMAKE_BUILD_TYPE', btype)
        return self.cmake

    def update_source_dir(self, *args):
        source_str = str(self.frm_source)
        if source_str:
            self.cmake.set_source_dir(source_str)
        else:
            self.cmake.set_source_dir('')

        if source_str and self.frm_in_place.is_enabled:
            build_dir = '"' + source_str.strip('"') + '/build/cmake"'
            self.cmake.add_opt('-B', build_dir)
            self.cmake.add_cmake_var('CMAKE_INSTALL_PREFIX', source_str)

        if (not source_str) and self.frm_in_place.is_enabled:
            self.cmake.rem_opt('-B')
            self.cmake.rem_opt('CMAKE_INSTALL_PREFIX')

        return self.cmake

    def update_build_dir(self, *args):
        build_str = str(self.frm_build)
        if build_str:
            self.cmake.add_opt('-B', build_str)
        else:
            self.cmake.rem_opt('-B')

        return self.cmake

    def update_install_dir(self, *args):
        install_str = str(self.frm_install)
        if install_str:
            self.cmake.add_cmake_var('CMAKE_INSTALL_PREFIX', install_str)
        else:
            self.cmake.rem_cmake_var('CMAKE_INSTALL_PREFIX')

        return self.cmake


class QMPicker(tk.Frame):
    """a frame of Quantum Mechanical options for CHARMM

    Attributes
    ----------
    parent : tk.Frame
        the containing widget
    cmake : CMakeLine
        the cmake command line to modify
    selection : tk.StringVar
        name of QM option chosen, set by radio buttons, one of qm_opts
    qm_opts : list of str list
        list of names of QM options
    radio_buttons : list of tk.Radiobutton
        a radio button for each of qm_opts to choose between them

    Methods
    -------
    create_widgets(selection, opts)
        initialize+pack radio btns from qm_opts, fills radio_buttons
    update_cmake()
        rem old qm opt and add new qm opt when radio buttons change
    """
    def __init__(self, parent, cmake, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)
        self.parent = parent
        self.cmake = cmake
        self.selection = tk.StringVar()
        self.qm_opts = ['g09', 'gamess', 'mndo97', 'qmmmsemi',
                        'qturbo', 'quantum', 'sccdftb', 'squantm']
        self.create_widgets(self.selection, self.qm_opts)
        rb_quantum = next(filter(lambda x: x['text'] == 'quantum',
                                 self.radio_buttons))
        rb_quantum.invoke()

    def create_widgets(self, selection, opts):
        self.radio_buttons = [
            tk.Radiobutton(self, text=opt, value=opt, var=selection,
                           command=self.update_cmake)
            for opt in opts
        ]
        for rb in self.radio_buttons:
            rb.pack(fill=tk.X)

    def update_cmake(self):
        for opt in self.qm_opts:
            self.cmake.rem_cmake_var(opt)

        selected = self.selection.get()
        self.cmake.add_cmake_var(selected)
        return(self.cmake)


class Messenger(tk.Frame):
    """text area with copy all button for messages from cmake and make

    Attributes
    ----------
    parent : tk.Frame
        the containing widget
    frm_title : tk.Frame
        a frame to hold the name and copy button
    lbl_title : tk.Label
        a label to go in frm_title to name the text area
    btn_copy : tk.Button
        a copy all button for the text area, lives in the frm_title
    txt_messages : tk.Text
        text area to display messages from cmake and make commands

    Methods
    -------
    copy()
        place contents of txt_messages into OS clipboard
    set_text(text)
        set contents of txt_messages to text, callback for ButtonBar
    get_text()
        get contents of txt_messages
    """
    def __init__(self, parent, root,
                 desc='Message Area',
                 *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)
        self.parent = parent

        self.frm_title = tk.Frame(self)
        self.lbl_title = tk.Label(self.frm_title, text=desc)
        self.lbl_title.pack(side=tk.LEFT)

        self.btn_copy = tk.Button(self.frm_title, text='copy',
                                  command=self.copy)
        self.btn_copy.pack(side=tk.RIGHT)
        self.frm_title.pack()

        self.txt_messages = tk.Text(self, state=tk.DISABLED)
        self.txt_messages.pack()

    def copy(self) -> str:
        root.clipboard_clear()
        text = self.get_text()
        root.clipboard_append(text)
        root.update()
        return text

    def set_text(self, text: str) -> str:
        old_text = self.get_text()
        self.txt_messages.config(state=tk.NORMAL)
        self.txt_messages.delete(1.0, tk.END)
        self.txt_messages.insert(tk.END, text)
        self.txt_messages.config(state=tk.DISABLED)
        return old_text

    def get_text(self) -> str:
        text = self.txt_messages.get(1.0, tk.END)
        return text


class ButtonBar(tk.Frame):
    """strip of buttons for actions related to OS and building project

    Attributes
    ----------
    parent : tk.Frame
        the containing widget
    cmake : CMakeLine
        the cmake command line to modify
    config_handler : function
        the func to run when the configure button is pressed
    make_handler : function
        the func to run when the make button is pressed
    btn_copy : tk.Button
        copy the cmake command line to the OS clipboard
    btn_config : tk.Button
        run the cmake command line in the build directory
    btn_make : tk.Button
        run the make command in the build directory
    btn_quit : tk.Button
        quit the whole app

    Methods
    -------
    create_widgets(root)
        init and pack all buttons
    copy()
        copy cmake command line to OS clipboard
    set_config_handler(func)
        set func to call when configure button is pressed
    config()
        formats cmake output, calls config_handler on config btn press
    set_make_handler(func)
        set func to call when make button is pressed
    make()
        formats cmake output, calls make_handler on make btn press
    """
    def __init__(self, parent, root, cmake, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)
        self.parent = parent
        self.cmake = cmake
        self.config_handler = None
        self.make_handler = None
        self.create_widgets(root)

    def create_widgets(self, root):
        self.btn_copy = tk.Button(self, text='copy cmake line',
                                  command=self.copy)
        self.btn_copy.pack(side=tk.LEFT)

        self.btn_config = tk.Button(self, text='configure',
                                    command=self.config)
        self.btn_config.pack(side=tk.LEFT)

        self.btn_make = tk.Button(self, text='make',
                                  command=self.make)
        self.btn_make.pack(side=tk.LEFT)

        self.btn_quit = tk.Button(self, text='quit',
                                  command=root.destroy)
        self.btn_quit.pack(side=tk.LEFT)

    def copy(self) -> str:
        root.clipboard_clear()
        text = self.cmake.command_line.get()
        root.clipboard_append(text)
        root.update()
        return text

    def set_config_handler(self, func):
        self.config_handler = func
        return func

    def config(self) -> str:
        output = self.cmake.config()
        text = (b'\n--- ERRORS and WARNINGS ---\n\n' + output['stderr']
                + b'\n--- CMAKE OUTPUT ---\n\n' + output['stdout'])
        self.config_handler(text)
        return text

    def set_make_handler(self, func):
        self.make_handler = func
        return func

    def make(self) -> str:
        output = self.cmake.make()
        text = (b'\n--- ERRORS and WARNINGS ---\n\n' + output['stderr']
                + b'\n--- CMAKE OUTPUT ---\n\n' + output['stdout'])
        self.make_handler(text)
        return text


class PathPicker(tk.Frame):
    """a label, entry, and browse button for selecting paths

    Attributes
    ----------
    parent : tk.Frame
        the containing widget
    command : function
        callback for when path changes
    want_dir : bool
        want a directory path and not a file
    var_path : tk.StringVar
        string path tied to entry widget
    enabled : bool
        is this widget enabled or disabled
    lbl_path : tk.Label
        label in front of entry to describe purpose to users
    ent_path : tk.Entry
        place to enter path by typing
    btn_browse : tk.Button
        press to get file finder dialog to fill ent_path

    Methods
    -------
    get_file()
        callback for browse button, fills var_path with user selection
    create_widgets(desc)
        init and pack widgets for label, path entry, and browse button
    disable()
        set entry and browse buttons to disabled
    enable()
        set entry and browse buttons to enabled
    is_enabled()
        have the widgets been disabled?
    set_path(path)
        sets the path in case this widget is tied to another one
    """
    def __init__(self, parent, desc, command=None, want_dir=False,
                 *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)
        self.parent = parent
        self.command = command
        self.want_dir = want_dir
        self.var_path = tk.StringVar()
        self.create_widgets(desc)
        self.enabled = True

    def get_file(self):
        if self.want_dir:
            path_str = filedialog.askdirectory()
        else:
            path_str = filedialog.askopenfilename()

        self.var_path.set(path_str)
        return path_str

    def create_widgets(self, desc):
        self.lbl_path = tk.Label(self, text=str(desc))
        self.ent_path = tk.Entry(self, textvariable=self.var_path)
        self.btn_browse = tk.Button(self, text='browse', command=self.get_file)

        self.lbl_path.pack(side=tk.LEFT)
        self.ent_path.pack(side=tk.LEFT)
        self.btn_browse.pack(side=tk.LEFT)

        self.var_path.trace('w', self.command)
        return self

    def __str__(self):
        path = self.var_path.get()
        if not self.is_enabled():
            path = ''

        if path:
            path = '"' + path + '"'

        return(path)

    def disable(self):
        self.ent_path.config(state='disabled')
        self.btn_browse.config(state='disable')
        self.enabled = False
        self.command()
        return self.enabled

    def enable(self):
        self.ent_path.config(state='normal')
        self.btn_browse.config(state='normal')
        self.enabled = True
        self.command()
        return self.enabled

    def is_enabled(self):
        return self.enabled

    def set_path(self, path):
        old_path = self.var_path.get()
        self.var_path.set(path)
        return old_path


class EnableToggle(tk.Frame):
    """check button that can enable/disable other widgets in the app

    Attributes
    ----------
    parent : tk.Frame
        the containing widget
    desc : str
        title to add to the check button
    cmake : CMakeLine
        the cmake command line to modify
    enable_widgets : list of tk.Frame
        list of widgets to enable when check button is checked
    disable_widgets : list of tk.Frame
        list of widgets to disable when check button is unchecked
    opt_name : str
        the cmake variable being toggled, if this is none, desc is used
    var_enabled : tk.BooleanVar
        tk variable to track state of check button
    chk_enable : tk.Checkbutton
        allow user to choose whether cmake var is set to ON or OFF

    Methods
    -------
    create_widgets(desc)
        init and pack the label and check button
    toggle()
        callback for check button, enables and disables the *_widgets
    is_enabled()
        query the state of the check button
    update_cmake()
        update the cmake variable associated with this widget
    """
    def __init__(self, parent, desc,
                 cmake,
                 enable_widgets=[], disable_widgets=[],
                 is_enabled=True, opt_name=None,
                 *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)
        self.parent = parent
        self.desc = desc
        self.cmake = cmake
        self.enable_widgets = enable_widgets
        self.disable_widgets = disable_widgets
        self.opt_name = opt_name
        self.var_enabled = tk.BooleanVar(value=is_enabled)

        self.create_widgets(desc)
        self.var_enabled.trace('w', self.update_cmake)

        # get the enable and disable widgets in the correct starting state
        self.toggle()
        self.toggle()

    def create_widgets(self, desc):
        if self.opt_name:
            text = self.desc
        else:
            text = 'Enable ' + self.desc

        self.chk_enable = tk.Checkbutton(self,
                                         text=text,
                                         onvalue=True, offvalue=False,
                                         variable=self.var_enabled,
                                         command=self.toggle)
        self.chk_enable.pack()

    def toggle(self):
        if self.is_enabled():
            for widget in self.enable_widgets:
                widget.enable()

            for widget in self.disable_widgets:
                widget.disable()

            self.var_enabled.set(True)
        else:
            for widget in self.enable_widgets:
                widget.disable()

            for widget in self.disable_widgets:
                widget.enable()

            self.var_enabled.set(False)

        return self.var_enabled.get()

    def is_enabled(self):
        return self.var_enabled.get()

    def update_cmake(self, *args):
        opt = 'OFF'
        if self.is_enabled():
            opt = 'ON'

        if self.opt_name:
            self.cmake.add_cmake_var(self.opt_name, opt)
        else:
            self.cmake.add_cmake_var(self.desc, opt)

        return self.cmake


class CompilersTab(tk.Frame):
    """options related to finding compilers C/CXX/Fortran

    Attributes
    ----------
    parent : tk.Frame
        the containing widget
    cmake : CMakeLine
        the cmake command line to modify
    frm_c : PathPicker
        allow user to input path to C compiler
    frm_cxx : PathPicker
        allow user to input path to C++ compiler
    frm_fort : PathPicker
        allow user to input path to Fortran compiler

    Methods
    -------
    create_widgets()
        init and pack the three PathPickers for C/C++/Fortran
    update_c()
        callback for the C compiler path, sets an env var
    update_cxx()
       callback for the C++ compiler path, sets an env var
    update_fort()
       callback for the Fortran compiler path, sets an env var
    """

    def __init__(self, parent, cmake, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)
        self.parent = parent
        self.cmake = cmake
        self.create_widgets()

    def create_widgets(self):
        self.frm_c = PathPicker(self, 'C', command=self.update_c)
        self.frm_c.pack()

        self.frm_cxx = PathPicker(self, 'C++', command=self.update_cxx)
        self.frm_cxx.pack()

        self.frm_fort = PathPicker(self, 'Fortran', command=self.update_fort)
        self.frm_fort.pack()

    def update_c(self, *args):
        c_comp = str(self.frm_c)
        if c_comp:
            self.cmake.add_env_var('CC', c_comp)
        else:
            self.cmake.rem_env_var('CC')

        return self.cmake

    def update_cxx(self, *args):
        cxx_comp = str(self.frm_cxx)
        if cxx_comp:
            self.cmake.add_env_var('CXX', cxx_comp)
        else:
            self.cmake.rem_env_var('CXX')

        return self.cmake

    def update_fort(self, *args):
        fort_comp = str(self.frm_fort)
        if fort_comp:
            self.cmake.add_env_var('FC', fort_comp)
        else:
            self.cmake.rem_env_var('FC')

        return(self.cmake)


class MPITab(tk.Frame):
    """options related to using and finding MPI for CHARMM

    Attributes
    ----------
    parent : tk.Frame
        the containing widget
    cmake : CMakeLine
        the cmake command line to modify
    frm_dir : PathPicker
        allow user to input path to MPI_HOME
    frm_comp : PathPicker
        allow user to input path to MPI_Fortran_COMPILER
    frm_enable : EnableToggle
        cmake var mpi set to ON or OFF per user input

    Methods
    -------
    create_widgets()
        inits and packs on screen widgets
    path2opt(path_picker, var_name)
        take a path and turn it into a cmake variable
    update_home()
        callback for setting cmake MPI_HOME
    update_compiler()
        callback for setting MPI compiler on the cmake command line
    """
    def __init__(self, parent, cmake, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)
        self.parent = parent
        self.cmake = cmake
        self.create_widgets()

    def create_widgets(self):
        self.frm_dir = PathPicker(self, 'MPI Base Directory',
                                  command=self.update_home, want_dir=True)
        self.frm_dir.pack()

        self.frm_comp = PathPicker(self, 'MPI Fortran Compiler',
                                   command=self.update_compiler)
        self.frm_comp.pack()

        self.frm_enable = EnableToggle(self, 'mpi', self.cmake,
                                       [self.frm_dir, self.frm_comp])
        self.frm_enable.pack()

    def path2opt(self, path_picker, var_name):
        path = str(path_picker)
        if path:
            self.cmake.add_cmake_var(var_name, path)
        else:
            self.cmake.rem_cmake_var(var_name)

        return self.cmake

    def update_home(self, *args):
        self.path2opt(self.frm_dir, 'MPI_HOME')

        return self.cmake

    def update_compiler(self, *args):
        self.path2opt(self.frm_comp, 'MPI_Fortran_COMPILER')

        return self.cmake

# BEGIN TODO AREA


class AccelTab(tk.Frame):
    def __init__(self, parent, cmake, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)
        self.parent = parent
        self.cmake = cmake
        self.create_widgets()

    def create_widgets(self):
        self.frm_domdec = EnableToggle(self, 'domdec', self.cmake)
        self.frm_domdec.pack()

        self.frm_openmm = EnableToggle(self, 'openmm', self.cmake)
        self.frm_openmm.pack()

        self.frm_domdec_gpu = EnableToggle(self, 'domdec_gpu', self.cmake,
                                           is_enabled=False)
        self.frm_domdec_gpu.pack()

        self.frm_repdstr = EnableToggle(self, 'repdstr', self.cmake,
                                        is_enabled=False)
        self.frm_repdstr.pack()


class ColfftTab(tk.Frame):
    def __init__(self, parent, cmake, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)
        self.parent = parent
        self.cmake = cmake
        self.var_fft_api = tk.StringVar(value='fftw')
        self.var_colfft = tk.BooleanVar(value=True)
        self.create_widgets()
        self.update_colfft()

    def create_widgets(self):
        self.chk_colfft = tk.Checkbutton(self, text='colfft',
                                         onvalue=True, offvalue=False,
                                         variable=self.var_colfft,
                                         command=self.update_colfft)
        self.chk_colfft.pack()

        self.rb_fftw = tk.Radiobutton(self, text='fftw',
                                      value='fftw',
                                      var=self.var_fft_api,
                                      command=self.update_fft_api)
        self.rb_fftw.pack()

        self.frm_fftw_home = PathPicker(self, 'FFTW_HOME',
                                        self.update_fftw_home, True)

        self.rb_mkl = tk.Radiobutton(self, text='mkl',
                                     value='mkl',
                                     var=self.var_fft_api,
                                     command=self.update_fft_api)
        self.rb_mkl.pack()

        self.frm_mkl_home = PathPicker(self, 'MKL_HOME',
                                       self.update_mkl_home, True)

        self.rb_fftw.invoke()

    def update_fftw_home(self, a=None, b=None, c=None):
        self.cmake.rem_env_var('FFTW_HOME')
        colfft_enabled = self.var_colfft.get()
        fftw_enabled = self.var_fft_api.get() == 'fftw'
        path = None
        if colfft_enabled and fftw_enabled:
            path = str(self.frm_fftw_home)
        else:
            self.cmake.rem_cmake_var('FFTW_HOME')

        if path:
            self.cmake.add_env_var('FFTW_HOME', path)

        return path

    def update_mkl_home(self, a=None, b=None, c=None):
        self.cmake.rem_env_var('MKL_HOME')
        colfft_enabled = self.var_colfft.get()
        mkl_enabled = self.var_fft_api.get() == 'mkl'
        path = None
        if colfft_enabled and mkl_enabled:
            path = str(self.frm_mkl_home)
        else:
            self.cmake.rem_cmake_var('MKL_HOME')

        if path:
            self.cmake.add_env_var('MKL_HOME', path)

        return path

    def update_fft_api(self):
        fft_api = self.var_fft_api.get()
        colfft_enabled = self.var_colfft.get()
        if colfft_enabled and (fft_api == 'fftw'):
            self.cmake.add_cmake_var('mkl', 'OFF')
            self.cmake.add_cmake_var('fftw')
            self.frm_mkl_home.pack_forget()
            self.frm_fftw_home.pack()
        elif colfft_enabled and (fft_api == 'mkl'):
            self.cmake.add_cmake_var('fftw', 'OFF')
            self.cmake.add_cmake_var('mkl')
            self.frm_fftw_home.pack_forget()
            self.frm_mkl_home.pack()
        elif not colfft_enabled:
            self.cmake.add_cmake_var('fftw', 'OFF')
            self.cmake.add_cmake_var('mkl', 'OFF')
            self.frm_fftw_home.pack_forget()
            self.frm_mkl_home.pack_forget()

        self.update_fftw_home()
        self.update_mkl_home()
        return fft_api

    def update_colfft(self):
        enabled = self.var_colfft.get()
        if enabled:  # TODO fft choice and path picker needs updating
            self.cmake.add_cmake_var('colfft')
            self.rb_fftw.pack()
            self.rb_mkl.pack()
        else:
            self.cmake.add_cmake_var('colfft', 'OFF')
            self.rb_fftw.pack_forget()
            self.rb_mkl.pack_forget()

        self.update_fft_api()
        return enabled


# END TODO AREA


class MainApplication(tk.Frame):
    """parent window for app

    Attributes
    ----------
    parent : tk.Frame
        the containing widget
    cmake : CMakeLine
        the cmake command line to modify
    buttonBar : ButtonBar
        buttons for executing actions related to cmake, also quit
    nb : ttk.Notebook
        a set of tabs for specifying options to cmake
    buildType : BuildOpts
        a tab of build options: source dir, debug, etc.
    compilers : CompilersTab
        a tab to specify compilers to use C/C++/Fortran
    mpi : MPITab
        a tab to specify options to help find MPI or turn MPI off
    qmPicker : QMPicker
        a tab to specify which QM package to use
    messenger : Messenger
        displays output of configure and make commands
    """
    def __init__(self, parent, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)
        self.parent = parent
        self.cmake = CMakeLine()

        self.buttonBar = ButtonBar(self, self.parent, self.cmake)
        self.buttonBar.pack(fill=tk.X)

        self.nb = ttk.Notebook(self)

        self.buildType = BuildOpts(self.nb, self.cmake)
        self.nb.add(self.buildType, text='build type')

        self.compilers = CompilersTab(self.nb, self.cmake)
        self.nb.add(self.compilers, text='compilers')

        self.mpi = MPITab(self.nb, self.cmake)
        self.nb.add(self.mpi, text='MPI')

        self.qmPicker = QMPicker(self.nb, self.cmake)
        self.nb.add(self.qmPicker, text='QM style')

        self.accelTab = AccelTab(self.nb, self.cmake)
        self.nb.add(self.accelTab, text='Special Accel')

        self.colfftTab = ColfftTab(self.nb, self.cmake)
        self.nb.add(self.colfftTab, text='FFT')

        self.nb.pack(expand=1, fill='both')

        self.messenger = Messenger(self, self.parent)
        self.messenger.pack()

        self.buttonBar.set_config_handler(self.messenger.set_text)
        self.buttonBar.set_make_handler(self.messenger.set_text)


if __name__ == '__main__':
    root = tk.Tk()
    root.title('Configure CHARMM')
    MainApplication(root).pack(side='top', fill='both', expand=True)
    root.mainloop()
