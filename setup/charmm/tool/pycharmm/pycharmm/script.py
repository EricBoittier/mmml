import numbers

import pycharmm.lingo
import pycharmm.select_atoms as atoms


class CommandScript:
    """ A class used to construct a command in CHARMM scripting language and then execute it.

    """
    def __init__(self, command, selection=None, **kwargs):
        """
        Parameters
        ----------
        command : str
            CHARMM scripting language command that will be run using pyCHARMM.
        selection : pycharmm.selectAtoms
        **kwargs : dict, optional
            Additional options relevant to the `command` being run.
        """
        self.command = str(command)
        self.selection = selection
        self.opts = dict()
        self.add_options(**kwargs)

    def add_options(self, **kwargs):
        """Add additional options relevant to the command being run.

        If value is of type `float`, `int` or `str` a "key value" 
        line is added to the command's script. 
        If it is of type `bool` and is True
        then a "key" line is added.
        """ 
        for k, v in kwargs.items():
            # must check for bools first because
            # a bool is also a numbers.Number but
            # numeric values must be handled differently
            if isinstance(v, bool):
                if v:
                    self.opts[k] = str(k) + ' -\n'
            # bytes added here to handle numpy.string_
            # numbers.Number should encompass float, int, and numpy.float64
            elif isinstance(v, (numbers.Number, bytes, str)):
                self.opts[k] = str(k) + ' ' + str(v) + ' -\n'
            else:
                message = 'invalid option {} = {}'
                raise ValueError(message.format(k, v))

    def _add_selection(self, selection):
        if not selection.is_stored():
            raise ValueError('store selection before adding to script')

        self.add_options(sele=selection.get_stored_name() + ' end')
        return

    def _remove_selection(self, selection):
        self.opts.pop('sel', None)
        selection.unstore()

    def create_script_string(self):
        script = self.command
        if self.opts.values():
            script += ' ' + ' '.join(self.opts.values())
            script = script.strip()
            if script.endswith('-'):
                script = script[:-1].strip()
            if not script.endswith('\n'):
                script += '\n'

        return script

    def run(self, append=''):
        """Execute the command.
        """
        if self.selection:
            selection = self.selection.get_selection()
            with atoms.SelectAtoms(selection=selection) as sel:
                self._add_selection(sel)
                script = self.create_script_string() + append
                pycharmm.lingo.charmm_script(script)
                self._remove_selection(sel)
        else:
            script = self.create_script_string() + append
            pycharmm.lingo.charmm_script(script)

        return self


class NonBondedScript(CommandScript):
    """A child of the `CommandScript` class for running CHARMM NBOND command.

    See https://academiccharmm.org/documentation/latest/nbonds for more details.
    """
    def __init__(self, **kwargs):
        super().__init__('nbonds', **kwargs)


class UpdateNonBondedScript(CommandScript):
    """A child of the `CommandScript` class for running CHARMM UPDATE command.

    See https://academiccharmm.org/documentation/latest/nbonds for more details.
    """
    # added by Yujin Wu (wyujin@umich.edu)
    
    def __init__(self, **kwargs):
        super().__init__('update', **kwargs)


class PatchScript(CommandScript):
    """A child of the `CommandScript` class for running CHARMM PATCH command.
    See https://academiccharmm.org/documentation/latest/struct#Patch for more details.
    """
    def __init__(self, **kwargs):
        super().__init__('patch', **kwargs)


class WriteScript(CommandScript):
    """A child of the `CommandScript` class for running CHARMM WRITE command.

    A multiline `title` can be provided.
    The title can be appended to later when the `run()` function is called using
    its `append` argument.
    """
    def __init__(self, filename, title='', **kwargs):
        super().__init__('write', name=filename, **kwargs)
        self.filename = filename
        self.title = ''
        if title:
            self.title = '* ' + title + '\n'
            self.title += '*\n'

    def run(self, append=''):
        to_append = ''
        append = append.strip()
        if append:
            to_append = append + '\n'

        if self.title:
            to_append += self.title

        super().run(append=to_append)
        return self


def script_factory(command='', required_args=None):
    """Create a CommandScript object for any CHARMM command.

    Parameters
    ----------
    command : str 
        the CHARMM command to be run (only the name)
    required_args : list[str]
        a list of argument names the command requires

    Returns
    -------
    UserScript : CommandScript
        A class with a run method to run the CHARMM command specified

    Examples
    --------
    >>> import pycharmm

    To run the GETEnegy command, one can do the following
    >>> NewGeteClass = pycharmm.script_factory('gete')
    >>> my_gete = NewGeteClass(...)
    >>> my_gete.run()
    
    """
    if required_args is None:
        required_args = []

    required_args = required_args[:]

    class UserScript(CommandScript):
        def __init__(self, *args, **kwargs):
            if len(args) > len(required_args):
                raise ValueError('Too many arguments passed.')

            # At this point all positional arguments are fine.
            for arg in required_args[len(args):]:
                if arg not in kwargs:
                    raise ValueError('Missing value for argument {}.'.format(arg))

            # At this point, all arguments have been passed either as
            # positional or keyword.
            if len(required_args) - len(args) != len(kwargs):
                raise ValueError('Too many arguments passed.')

            for k, v in zip(required_args, args):
                kwargs[k] = v

            super().__init__(command, **kwargs)

    return UserScript
