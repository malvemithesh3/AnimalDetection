import smtplib

def sendmail(email,msg,loc):
    TO = email
    SUBJECT = 'Prediction:'
    TEXT ='Message:'+msg+'location:'+loc
     
    print(TEXT)
    # Gmail Sign In
    gmail_sender = "animaldetection98@gmail.com"
    gmail_passwd = "wlfuwbipnfmehpmr"

    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.ehlo()
    server.starttls()
    server.login(gmail_sender, gmail_passwd)

    BODY = '\r\n'.join(['To: %s' % TO,
                        'From: %s' % gmail_sender,
                        'Subject: %s' % SUBJECT,
                        '', TEXT])

    try:
        server.sendmail(gmail_sender, [TO], BODY)
        print ('email sent')
    except:
        print ('error sending mail')

    server.quit()
