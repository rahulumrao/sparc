def banner():
    from sparc.src.utils.logger import SparcLog

    banner_text = """
         ######  ########     ###    ########   ######
        ##    ## ##     ##   ## ##   ##     ## ##    ##
        ##       ##     ##  ##   ##  ##     ## ##
         ######  ########  ##     ## ########  ##
              ## ##        ######### ##   ##   ##
        ##    ## ##        ##     ## ##    ##  ##    ##
         ######  ##        ##     ## ##     ##  ######
         --v0.2
"""
    SparcLog(banner_text)


if __name__ == "__main__":
    banner()
