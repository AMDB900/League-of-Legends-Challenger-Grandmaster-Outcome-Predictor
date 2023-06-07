from flask import Flask, render_template, request
from flask_wtf import FlaskForm
from wtforms import SelectField, SubmitField
from wtforms.validators import DataRequired
from flask_wtf.csrf import CSRFProtect
import secrets
import sklearn.neural_network
import sklearn.preprocessing
import pickle
import pandas as pd
import torch
import numpy as np

champion_list = [
    ('Aatrox', 'Aatrox'),
    ('Ahri', 'Ahri'),
    ('Akali', 'Akali'),
    ('Akshan', 'Akshan'),
    ('Alistar', 'Alistar'),
    ('Amumu', 'Amumu'),
    ('Anivia', 'Anivia'),
    ('Annie', 'Annie'),
    ('Aphelios', 'Aphelios'),
    ('Ashe', 'Ashe'),
    ('AurelionSol', 'AurelionSol'),
    ('Azir', 'Azir'),
    ('Bard', 'Bard'),
    ('Belveth', 'Belveth'),
    ('Blitzcrank', 'Blitzcrank'),
    ('Brand', 'Brand'),
    ('Braum', 'Braum'),
    ('Caitlyn', 'Caitlyn'),
    ('Camille', 'Camille'),
    ('Cassiopeia', 'Cassiopeia'),
    ('Chogath', 'Chogath'),
    ('Corki', 'Corki'),
    ('Darius', 'Darius'),
    ('Diana', 'Diana'),
    ('DrMundo', 'DrMundo'),
    ('Draven', 'Draven'),
    ('Ekko', 'Ekko'),
    ('Elise', 'Elise'),
    ('Evelynn', 'Evelynn'),
    ('Ezreal', 'Ezreal'),
    ('FiddleSticks', 'FiddleSticks'),
    ('Fiora', 'Fiora'),
    ('Fizz', 'Fizz'),
    ('Galio', 'Galio'),
    ('Gangplank', 'Gangplank'),
    ('Garen', 'Garen'),
    ('Gnar', 'Gnar'),
    ('Gragas', 'Gragas'),
    ('Graves', 'Graves'),
    ('Gwen', 'Gwen'),
    ('Hecarim', 'Hecarim'),
    ('Heimerdinger', 'Heimerdinger'),
    ('Illaoi', 'Illaoi'),
    ('Irelia', 'Irelia'),
    ('Ivern', 'Ivern'),
    ('Janna', 'Janna'),
    ('JarvanIV', 'JarvanIV'),
    ('Jax', 'Jax'),
    ('Jayce', 'Jayce'),
    ('Jhin', 'Jhin'),
    ('Jinx', 'Jinx'),
    ('KSante', 'KSante'),
    ('Kaisa', 'Kaisa'),
    ('Kalista', 'Kalista'),
    ('Karma', 'Karma'),
    ('Karthus', 'Karthus'),
    ('Kassadin', 'Kassadin'),
    ('Katarina', 'Katarina'),
    ('Kayle', 'Kayle'),
    ('Kayn', 'Kayn'),
    ('Kennen', 'Kennen'),
    ('Khazix', 'Khazix'),
    ('Kindred', 'Kindred'),
    ('Kled', 'Kled'),
    ('KogMaw', 'KogMaw'),
    ('Leblanc', 'Leblanc'),
    ('LeeSin', 'LeeSin'),
    ('Leona', 'Leona'),
    ('Lillia', 'Lillia'),
    ('Lissandra', 'Lissandra'),
    ('Lucian', 'Lucian'),
    ('Lulu', 'Lulu'),
    ('Lux', 'Lux'),
    ('Malphite', 'Malphite'),
    ('Malzahar', 'Malzahar'),
    ('Maokai', 'Maokai'),
    ('MasterYi', 'MasterYi'),
    ('MissFortune', 'MissFortune'),
    ('MonkeyKing', 'Wukong'),
    ('Mordekaiser', 'Mordekaiser'),
    ('Morgana', 'Morgana'),
    ('Nami', 'Nami'),
    ('Nasus', 'Nasus'),
    ('Nautilus', 'Nautilus'),
    ('Neeko', 'Neeko'),
    ('Nidalee', 'Nidalee'),
    ('Nilah', 'Nilah'),
    ('Nocturne', 'Nocturne'),
    ('Nunu', 'Nunu'),
    ('Olaf', 'Olaf'),
    ('Orianna', 'Orianna'),
    ('Ornn', 'Ornn'),
    ('Pantheon', 'Pantheon'),
    ('Poppy', 'Poppy'),
    ('Pyke', 'Pyke'),
    ('Qiyana', 'Qiyana'),
    ('Quinn', 'Quinn'),
    ('Rakan', 'Rakan'),
    ('Rammus', 'Rammus'),
    ('RekSai', 'RekSai'),
    ('Rell', 'Rell'),
    ('Renata', 'Renata'),
    ('Renekton', 'Renekton'),
    ('Rengar', 'Rengar'),
    ('Riven', 'Riven'),
    ('Rumble', 'Rumble'),
    ('Ryze', 'Ryze'),
    ('Samira', 'Samira'),
    ('Sejuani', 'Sejuani'),
    ('Senna', 'Senna'),
    ('Seraphine', 'Seraphine'),
    ('Sett', 'Sett'),
    ('Shaco', 'Shaco'),
    ('Shen', 'Shen'),
    ('Shyvana', 'Shyvana'),
    ('Singed', 'Singed'),
    ('Sion', 'Sion'),
    ('Sivir', 'Sivir'),
    ('Skarner', 'Skarner'),
    ('Sona', 'Sona'),
    ('Soraka', 'Soraka'),
    ('Swain', 'Swain'),
    ('Sylas', 'Sylas'),
    ('Syndra', 'Syndra'),
    ('TahmKench', 'TahmKench'),
    ('Taliyah', 'Taliyah'),
    ('Talon', 'Talon'),
    ('Taric', 'Taric'),
    ('Teemo', 'Teemo'),
    ('Thresh', 'Thresh'),
    ('Tristana', 'Tristana'),
    ('Trundle', 'Trundle'),
    ('Tryndamere', 'Tryndamere'),
    ('TwistedFate', 'TwistedFate'),
    ('Twitch', 'Twitch'),
    ('Udyr', 'Udyr'),
    ('Urgot', 'Urgot'),
    ('Varus', 'Varus'),
    ('Vayne', 'Vayne'),
    ('Veigar', 'Veigar'),
    ('Velkoz', 'Velkoz'),
    ('Vex', 'Vex'),
    ('Vi', 'Vi'),
    ('Viego', 'Viego'),
    ('Viktor', 'Viktor'),
    ('Vladimir', 'Vladimir'),
    ('Volibear', 'Volibear'),
    ('Warwick', 'Warwick'),
    ('Xayah', 'Xayah'),
    ('Xerath', 'Xerath'),
    ('XinZhao', 'XinZhao'),
    ('Yasuo', 'Yasuo'),
    ('Yone', 'Yone'),
    ('Yorick', 'Yorick'),
    ('Yuumi', 'Yuumi'),
    ('Zac', 'Zac'),
    ('Zed', 'Zed'),
    ('Zeri', 'Zeri'),
    ('Ziggs', 'Ziggs'),
    ('Zilean', 'Zilean'),
    ('Zoe', 'Zoe'),
    ('Zyra', 'Zyra')
]
class Myform(FlaskForm):
    model_loading = SelectField('ML Model', choices=[
        ('logistic_regression', 'Logistic Regression'),
        ('mlp', 'MLP'),
        ('random_forest', 'Random Forest'),
        ('knn', 'kNN')
    ], validators=[DataRequired()])
    top = SelectField('Top', choices=champion_list, validators=[DataRequired()])
    jungle = SelectField('Jungle', choices=champion_list, validators=[DataRequired()])
    mid = SelectField('Mid', choices=champion_list, validators=[DataRequired()])
    bot = SelectField('Bottom', choices=champion_list, validators=[DataRequired()])
    support = SelectField('Support', choices=champion_list, validators=[DataRequired()])
    enemy_top = SelectField('Enemy Top', choices=champion_list, validators=[DataRequired()])
    enemy_jungle = SelectField('Enemy Jungle', choices=champion_list, validators=[DataRequired()])
    enemy_mid = SelectField('Enemy Mid', choices=champion_list, validators=[DataRequired()])
    enemy_bot = SelectField('Enemy Bottom', choices=champion_list, validators=[DataRequired()])
    enemy_support = SelectField('Enemy Support', choices=champion_list, validators=[DataRequired()])
    submit = SubmitField('Predict')

secret_key = secrets.token_hex(32)
app = Flask(__name__)
app.secret_key = secret_key
csrf = CSRFProtect(app)
@app.route('/', methods=['GET', 'POST'])
def form_input():
    form = Myform()
    output = None  # Initialize the output variable
    prob_out = None  # Initialize the prob_out variable
    if form.validate_on_submit():
        top = form.top.data
        mid = form.mid.data
        bot = form.bot.data
        jungle = form.jungle.data
        support = form.support.data
        enemy_top = form.enemy_top.data
        enemy_mid = form.enemy_mid.data
        enemy_bot = form.enemy_bot.data
        enemy_jungle = form.enemy_jungle.data
        enemy_support = form.enemy_support.data
        n_model = form.model_loading.data
        print(n_model)
        encoding_scheme = pickle.load(open(app.root_path + '/encoding_scheme.pkl', 'rb'))
        new_input = pd.DataFrame(columns=encoding_scheme)
        row_with_zeroes = pd.Series(0, index=encoding_scheme)

        # Add the row to the new_input DataFrame
        new_input = pd.concat([new_input, row_with_zeroes.to_frame().T], ignore_index=True)
        input_dict = {
            'Enemy Top': enemy_top,
            'Enemy Jungle': enemy_jungle,
            'Enemy Mid': enemy_mid,
            'Enemy Bot': enemy_bot,
            'Enemy Support': enemy_support,
            'Top': top,
            'Jungle': jungle,
            'Mid': mid,
            'Bot': bot,
            'Support': support
        }
        df = pd.DataFrame(input_dict, index=[0])
        encoded_input = pd.get_dummies(df)
        intersect = new_input.columns.intersection(encoded_input.columns)
        new_input[intersect] = 1

        if torch.cuda.is_available():
            model = torch.load(app.root_path + '/fnn_num_model.pth')
        else:
            model = torch.load(app.root_path + '/fnn_num_model.pth', map_location=torch.device('cpu'))
        model.eval()
        input_tensor = torch.tensor(np.array(new_input, dtype=np.float32), dtype=torch.float32)

        if torch.cuda.is_available():
            input_tensor = input_tensor.cuda()
        with torch.no_grad():
            output = model(input_tensor)

        scaler = pickle.load(open(app.root_path + '/scaler.pkl', 'rb'))
        post_out = np.round(scaler.inverse_transform(output.cpu().numpy())).astype(int)

        # Replace model here if needed
        model_file = '/random_forest_model.pkl'
        if(n_model == 'logistic_regression'):
            model_file = '/logistic_regression_model.pkl' # Change to logistic Regression model
        if(n_model == 'mlp'):
            model_file = '/mpl_model.pkl' # Change to MLP Model
        if(n_model == 'random_forest'):
            model_file = '/random_forest_model.pkl'
        if(n_model == 'knn'):
            model_file = '/kNN_model.pkl' # Change to kNN Model

        loaded_model = pickle.load(open(app.root_path + model_file, 'rb'))
        output = loaded_model.predict(post_out)[0]
        prob_out = loaded_model.predict_proba(post_out)[:, 1]
        if (output):
            output = "Victory"
        else:
            output = "Defeat"
    return render_template('form.html', form=form, output=output, prob_out=prob_out)

if __name__ == "__main__":
    app.run()