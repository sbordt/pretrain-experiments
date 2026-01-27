import paramiko
import os

# username
username = "sbordt10"

# galvani
galvani_hostname = "galvani-login.mlcloud.uni-tuebingen.de"
galvani_port = 2221

# ferranti
ferranti_hostname = "134.2.168.202"
ferranti_port = 22


def exec(command,
         hostname,
         port,
         username,
         return_output=False,
         print_output=True):
    try:
        # Initialize the SSH client
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(hostname, port, username, "")

        # execute the command
        stdin, stdout, stderr = client.exec_command(command)
        stdout = stdout.read().decode()
        stderr = stderr.read().decode()
        if print_output:
            print(stdout)
            print(stderr)
            
        # Close the SSH client
        client.close()
    except Exception as e:
        print(f"Failed to execute ssh command {command}. Exception: {e}")

    if return_output:
        return stdout


def galvani_exec(command, return_output=False, print_output=True):
    return exec(command, galvani_hostname, galvani_port, username, return_output, print_output)


def ferranti_exec(command, return_output=False, print_output=True):
    return exec(command, ferranti_hostname, ferranti_port, username, return_output, print_output)


def get_most_recent_files(sftp, directory, num_files, extension='.err'):
    """Retrieve the n most recent files from a directory."""
    files = sftp.listdir_attr(directory)
    files = sorted(files, key=lambda x: x.st_mtime, reverse=True)
    # only keep .err and .out files
    files = [file for file in files if file.filename.endswith('.err') or file.filename.endswith('.out')]
    return files[:num_files]


def print_logs(logs_dir,
               hostname,
               port,
               username,
               num_files):
    try:
        # Initialize the SSH client
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(hostname, port, username, "")

        # Start SFTP session
        sftp = client.open_sftp()

        # Get the most recent files
        recent_files = get_most_recent_files(sftp, logs_dir, num_files)
        
        # Print the content of the most recent files
        for file_attr in recent_files:
            file_path = os.path.join(logs_dir, file_attr.filename)
            print(f"Content of {file_path}:")
            with sftp.file(file_path, 'r') as f:
                file = f.read()
                # convert binary to string
                file = file.decode()
                print(file)
                print("="*80)  # Separator between file contents
                print("="*80)  # Separator between file contents

        # Close the SFTP session and SSH client
        sftp.close()
        client.close()

    except Exception as e:
        print(f"An error occurred: {e}")


def galvani_print_logs(path, num_files=5):
    print_logs(path, galvani_hostname, galvani_port, username, num_files)


def ferranti_print_logs(path, num_files=5):
    print_logs(path, ferranti_hostname, ferranti_port, username, num_files)