# RexGraph root build definition (meson-python backend).
# Compiles all Cython extensions to shared objects.

# Use make install-core for a regular build or make install-dev for development.
# Tests live in rexgraph/tests; run them against the selected installed runtime.
project(
    'rexgraph',
    'c', 'cython',
    # Keep Meson's reported version aligned with pyproject.toml, which supplies
    # the distribution metadata used by meson-python.
    version: '1.4.0',
    license: 'Apache-2.0',
    meson_version: '>=1.2.2',
    default_options: [
        'c_std=c11',
        'buildtype=release',
        'b_ndebug=if-release',
        'b_lto=true',
        'warning_level=2',
    ],
)

py = import('python').find_installation(pure: false)

cc = meson.get_compiler('c')
c_args = []
link_args = []
# Portable builds avoid CPU-specific instructions and relaxed float rules.
# Select -Dnative=true for a build intended for the current machine. On GCC and
# Clang this enables -march=native and -ffast-math, which can require newer CPU
# instructions and change NaN, infinity and rounding behavior.
if cc.get_id() == 'gcc' or cc.get_id() == 'clang'
    if get_option('native')
        c_args += ['-march=native', '-ffast-math']
    endif
    if get_option('buildtype') == 'release'
        c_args += ['-DNDEBUG']
    endif
elif cc.get_id() == 'msvc'
    c_args += ['/O2']
    if get_option('native')
        c_args += ['/fp:fast']
    endif
endif

# Query NumPy through the build interpreter so Cython declarations and C headers
# come from the same installation. A global dependency lookup can select another
# interpreter's headers and produce incompatible C API declarations.
# Pass the path as a compiler argument: a virtual environment inside the checkout
# puts NumPy under the source tree, which include_directories() refuses as an
# absolute include path. Use /I for MSVC and -I for the other compilers.
numpy_inc = run_command(py, '-c', 'import numpy; print(numpy.get_include())',
                        check: true).stdout().strip()
numpy_dep = declare_dependency(
    compile_args: [(cc.get_id() == 'msvc' ? '/I' : '-I') + numpy_inc])
omp_dep = dependency('openmp', required: get_option('openmp'))
m_dep = cc.find_library('m', required: false)
mvec_dep = cc.find_library('mvec', required: false)

cython_args = [
    '--fast-fail',
    '-X', 'boundscheck=False',
    '-X', 'wraparound=False',
    '-X', 'cdivision=True',
    '-X', 'language_level=3',
]
add_project_arguments(cython_args, language: 'cython')

subdir('rexgraph')
