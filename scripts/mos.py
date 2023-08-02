# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


"""
To run this script, from the root of the repo. Make sure to have Flask installed

    FLASK_DEBUG=1 FLASK_APP=scripts.mos flask run -p 4567
    # or if you have gunicorn
    gunicorn -w 4 -b 127.0.0.1:8895 -t 120 'scripts.mos:app'  --access-logfile -

"""
from collections import defaultdict
from functools import wraps
from hashlib import sha1
import json
import math
from pathlib import Path
import random
import typing as tp

from flask import Flask, redirect, render_template, request, session, url_for

from audiocraft import train
from audiocraft.utils.samples.manager import get_samples_for_xps


SAMPLES_PER_PAGE = 8
MAX_RATING = 5
storage = Path(train.main.dora.dir / 'mos_storage')
storage.mkdir(exist_ok=True)
surveys = storage / 'surveys'
surveys.mkdir(exist_ok=True)
magma_root = Path(train.__file__).parent.parent
app = Flask('mos', static_folder=str(magma_root / 'scripts/static'),
            template_folder=str(magma_root / 'scripts/templates'))
app.secret_key = b'audiocraft makes the best songs'


def normalize_path(path: Path):
    """Just to make path a bit nicer, make them relative to the Dora root dir.
    """
    path = path.resolve()
    dora_dir = train.main.dora.dir.resolve() / 'xps'
    return path.relative_to(dora_dir)


def get_full_path(normalized_path: Path):
    """Revert `normalize_path`.
    """
    return train.main.dora.dir.resolve() / 'xps' / normalized_path


def get_signature(xps: tp.List[str]):
    """Return a signature for a list of XP signatures.
    """
    return sha1(json.dumps(xps).encode()).hexdigest()[:10]


def ensure_logged(func):
    """Ensure user is logged in.
    """
    @wraps(func)
    def _wrapped(*args, **kwargs):
        user = session.get('user')
        if user is None:
            return redirect(url_for('login', redirect_to=request.url))
        return func(*args, **kwargs)
    return _wrapped


@app.route('/login', methods=['GET', 'POST'])
def login():
    """Login user if not already, then redirect.
    """
    user = session.get('user')
    if user is None:
        error = None
        if request.method == 'POST':
            user = request.form['user']
            if not user:
                error = 'User cannot be empty'
        if user is None or error:
            return render_template('login.html', error=error)
    assert user
    session['user'] = user
    redirect_to = request.args.get('redirect_to')
    if redirect_to is None:
        redirect_to = url_for('index')
    return redirect(redirect_to)


@app.route('/', methods=['GET', 'POST'])
@ensure_logged
def index():
    """Offer to create a new study.
    """
    errors = []
    if request.method == 'POST':
        xps_or_grids = [part.strip() for part in request.form['xps'].split()]
        xps = set()
        for xp_or_grid in xps_or_grids:
            xp_path = train.main.dora.dir / 'xps' / xp_or_grid
            if xp_path.exists():
                xps.add(xp_or_grid)
                continue
            grid_path = train.main.dora.dir / 'grids' / xp_or_grid
            if grid_path.exists():
                for child in grid_path.iterdir():
                    if child.is_symlink():
                        xps.add(child.name)
                continue
            errors.append(f'{xp_or_grid} is neither an XP nor a grid!')
        assert xps or errors
        blind = 'true' if request.form.get('blind') == 'on' else 'false'
        xps = list(xps)
        if not errors:
            signature = get_signature(xps)
            manifest = {
                'xps': xps,
            }
            survey_path = surveys / signature
            survey_path.mkdir(exist_ok=True)
            with open(survey_path / 'manifest.json', 'w') as f:
                json.dump(manifest, f, indent=2)
            return redirect(url_for('survey', blind=blind, signature=signature))
    return render_template('index.html', errors=errors)


@app.route('/survey/<signature>', methods=['GET', 'POST'])
@ensure_logged
def survey(signature):
    success = request.args.get('success', False)
    seed = int(request.args.get('seed', 4321))
    blind = request.args.get('blind', 'false') in ['true', 'on', 'True']
    exclude_prompted = request.args.get('exclude_prompted', 'false') in ['true', 'on', 'True']
    exclude_unprompted = request.args.get('exclude_unprompted', 'false') in ['true', 'on', 'True']
    max_epoch = int(request.args.get('max_epoch', '-1'))
    survey_path = surveys / signature
    assert survey_path.exists(), survey_path

    user = session['user']
    result_folder = survey_path / 'results'
    result_folder.mkdir(exist_ok=True)
    result_file = result_folder / f'{user}_{seed}.json'

    with open(survey_path / 'manifest.json') as f:
        manifest = json.load(f)

    xps = [train.main.get_xp_from_sig(xp) for xp in manifest['xps']]
    names, ref_name = train.main.get_names(xps)

    samples_kwargs = {
        'exclude_prompted': exclude_prompted,
        'exclude_unprompted': exclude_unprompted,
        'max_epoch': max_epoch,
    }
    matched_samples = get_samples_for_xps(xps, epoch=-1, **samples_kwargs)  # fetch latest epoch
    models_by_id = {
        id: [{
            'xp': xps[idx],
            'xp_name': names[idx],
            'model_id': f'{xps[idx].sig}-{sample.id}',
            'sample': sample,
            'is_prompted': sample.prompt is not None,
            'errors': [],
        } for idx, sample in enumerate(samples)]
        for id, samples in matched_samples.items()
    }
    experiments = [
        {'xp': xp, 'name': names[idx], 'epoch': list(matched_samples.values())[0][idx].epoch}
        for idx, xp in enumerate(xps)
    ]

    keys = list(matched_samples.keys())
    keys.sort()
    rng = random.Random(seed)
    rng.shuffle(keys)
    model_ids = keys[:SAMPLES_PER_PAGE]

    if blind:
        for key in model_ids:
            rng.shuffle(models_by_id[key])

    ok = True
    if request.method == 'POST':
        all_samples_results = []
        for id in model_ids:
            models = models_by_id[id]
            result = {
                'id': id,
                'is_prompted': models[0]['is_prompted'],
                'models': {}
            }
            all_samples_results.append(result)
            for model in models:
                rating = request.form[model['model_id']]
                if rating:
                    rating = int(rating)
                    assert rating <= MAX_RATING and rating >= 1
                    result['models'][model['xp'].sig] = rating
                    model['rating'] = rating
                else:
                    ok = False
                    model['errors'].append('Please rate this model.')
        if ok:
            result = {
                'results': all_samples_results,
                'seed': seed,
                'user': user,
                'blind': blind,
                'exclude_prompted': exclude_prompted,
                'exclude_unprompted': exclude_unprompted,
            }
            print(result)
            with open(result_file, 'w') as f:
                json.dump(result, f)
            seed = seed + 1
            return redirect(url_for(
                'survey', signature=signature, blind=blind, seed=seed,
                exclude_prompted=exclude_prompted, exclude_unprompted=exclude_unprompted,
                max_epoch=max_epoch, success=True))

    ratings = list(range(1, MAX_RATING + 1))
    return render_template(
        'survey.html', ratings=ratings, blind=blind, seed=seed, signature=signature, success=success,
        exclude_prompted=exclude_prompted, exclude_unprompted=exclude_unprompted, max_epoch=max_epoch,
        experiments=experiments, models_by_id=models_by_id, model_ids=model_ids, errors=[],
        ref_name=ref_name, already_filled=result_file.exists())


@app.route('/audio/<path:path>')
def audio(path: str):
    full_path = Path('/') / path
    assert full_path.suffix in [".mp3", ".wav"]
    return full_path.read_bytes(), {'Content-Type': 'audio/mpeg'}


def mean(x):
    return sum(x) / len(x)


def std(x):
    m = mean(x)
    return math.sqrt(sum((i - m)**2 for i in x) / len(x))


@app.route('/results/<signature>')
@ensure_logged
def results(signature):

    survey_path = surveys / signature
    assert survey_path.exists(), survey_path
    result_folder = survey_path / 'results'
    result_folder.mkdir(exist_ok=True)

    # ratings per model, then per user.
    ratings_per_model = defaultdict(list)
    users = []
    for result_file in result_folder.iterdir():
        if result_file.suffix != '.json':
            continue
        with open(result_file) as f:
            results = json.load(f)
        users.append(results['user'])
        for result in results['results']:
            for sig, rating in result['models'].items():
                ratings_per_model[sig].append(rating)

    fmt = '{:.2f}'
    models = []
    for model in sorted(ratings_per_model.keys()):
        ratings = ratings_per_model[model]

        models.append({
            'sig': model,
            'samples': len(ratings),
            'mean_rating': fmt.format(mean(ratings)),
            # the value 1.96 was probably chosen to achieve some
            # confidence interval assuming gaussianity.
            'std_rating': fmt.format(1.96 * std(ratings) / len(ratings)**0.5),
        })
    return render_template('results.html', signature=signature, models=models, users=users)
