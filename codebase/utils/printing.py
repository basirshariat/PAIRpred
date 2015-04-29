__author__ = 'basir'
from colorama import *
import sys


def print_info(message):
    print Fore.GREEN + "{0}".format(message) + Fore.RESET


def print_warning(message):
    print Fore.YELLOW + "{0}".format(message) + Fore.RESET


def print_error(message):
    print Fore.RED + "{0}".format(message) + Fore.RESET


def print_special(message):
    print Fore.CYAN + "{0}".format(message) + Fore.RESET


def print_info_nn(message):
    sys.stdout.write(Fore.GREEN + "{0}".format(message) + Fore.RESET)