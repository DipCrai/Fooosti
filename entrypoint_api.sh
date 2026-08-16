#!/bin/bash

ORIGINALDIR=/content/app
# Use predefined DATADIR if it is defined
[[ x"${DATADIR}" == "x" ]] && DATADIR=/content/data

# Make persistent dir from original dir
function mklink () {
	mkdir -p $DATADIR/$1
	ln -s $DATADIR/$1 $ORIGINALDIR
}

# Copy old files from import dir
function import () {
	(test -d /import/$1 && cd /import/$1 && cp -Rpn . $DATADIR/$1/)
}

cd $ORIGINALDIR

# models
mklink models
# Copy original files (skip the upstream template's nested 'models' subdir)
(cd $ORIGINALDIR/models.org && for f in *; do [ "$f" = "models" ] && continue; cp -Rpn "$f" $ORIGINALDIR/models/ 2>/dev/null; done)
# Drop the empty "put_*_here" placeholder markers from the template
rm -f $ORIGINALDIR/models/*/put_*_here $ORIGINALDIR/models/put_*_here 2>/dev/null
# Import old files
import models

# outputs
mklink outputs
# Import old files
import outputs

# Start Fooosti: torch-free API main (8890) and optional WebUI (7865) via a
# single queue manager + shared generation worker. Disable WebUI with
# FOOOSTI_WEBUI=0, or run only one server with --only-api/--only-webui.
echo '[Fooosti] starting launcher (api=8890 webui=7865, webui='"${FOOOSTI_WEBUI:-1}"')'

# Forward termination signals to the python launcher so the worker gets a
# chance to release VRAM/RAM on docker stop instead of being SIGKILLed.
_term() { kill -TERM "$child" 2>/dev/null; }
trap _term TERM INT
python fooosti.py $* &
child=$!
wait "$child"
