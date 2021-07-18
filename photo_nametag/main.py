from photo_nametag.face_cut import face_cut_main 
from photo_nametag.make_model import make_model_main
from photo_nametag.name_tag import name_tag_main
import click 

@click.group(invoke_without_command=True)
@click.pass_context
def cli(ctx):
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())

@cli.command()
def face_cut():
    '''
    モデルを作る用に顔の切り取り
    '''
    face_cut_main()

@cli.command()
def make_model():
    '''
    モデルを制作
    '''
    make_model_main()


@cli.command()
@click.option('-per',
             type=int,
             default=30,
             show_default=True,
             help='顔の判定にて指定した確率以上の人をタグ付け及び登録')
def name_tag(per):
    '''
    作ったモデルで写真にタグ付け
    '''
    name_tag_main(per)


