from twilio.rest import Client
import keys
result_path='to see accident image visit this directory\n"D:\\barsh\\Documents\\vscodefiles\\PROJECTS\\ML\\ACCIDENT_DITECTED_SYS\\RES"'
def send_msg():
    client = Client(keys.account_sid, keys.auth_token)

    message= client.messages.create (
    body=f" Accident Detected::\n{result_path}",
    from_=keys.twilio_num,
    to= keys.target_num
    )
send_msg()
