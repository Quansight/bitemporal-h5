$PROJECT = $GITHUB_REPO = 'bitemporal-h5'
$GITHUB_ORG = "Quansight"
$ACTIVITIES = ['authors', 'version_bump', 'changelog',
               'tag', 'push_tag', 'ghrelease', 'pypi',
               'conda_forge',
              ]
$PYPI_SIGN = False

#
# Version bumping
#
$VERSION_BUMP_PATTERNS = [
    ('setup.py', r'version\s*=.*,', 'version="$VERSION",'),
    ('bth5/__init__.py', r'__version__\s*=.*', '__version__ = "$VERSION"'),
    ('docs/conf.py', r'__version__\s*=.*', '__version__ = "$VERSION"'),
]

#
# Changelog
#
$CHANGELOG_FILENAME = 'CHANGELOG.md'
$CHANGELOG_TEMPLATE = 'TEMPLATE.md'
$CHANGELOG_PATTERN = "<!-- current developments -->"
$CHANGELOG_HEADER = """
<!-- current developments -->

## v$VERSION
"""
