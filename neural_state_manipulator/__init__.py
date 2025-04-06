"""
Neural State Manipulator - A tool for manipulating the internal neural activations of LLMs.

This package provides tools for recording and manipulating the internal neural 
activations of language models to control their generation behavior.
"""

from .manipulator import NeuralStateManipulator
from .utils import list_manipulable_layers, hook_manager

__version__ = '0.1.0'
__all__ = ['NeuralStateManipulator', 'list_manipulable_layers', 'hook_manager']
